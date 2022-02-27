/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if HYBRIDBACKEND_TENSORFLOW

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <vector>

#include "hybridbackend/cpp/tensorflow/cuda/cast.h"
#include "hybridbackend/cpp/tensorflow/nccl/nccl_comm.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

REGISTER_OP("AlltoallvWithNcclComm")
    .Output("output: T")
    .Output("output_sizes: int64")
    .Input("handle: resource")
    .Input("input: T")
    .Input("input_sizes: int64")
    .Attr("common_shape: shape = {}")
    .Attr("wire_dtype_for_float: {half, float}")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      PartialTensorShape common_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("common_shape", &common_shape));
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(
          c->MakeShapeFromPartialTensorShape(common_shape, &shape));
      TF_RETURN_IF_ERROR(c->Concatenate(
          c->Vector(shape_inference::InferenceContext::kUnknownDim), shape,
          &shape));
      c->set_output(0, shape);
      return Status::OK();
    })
    .Doc(R"doc(
AllToAllv using a NCCL communicator with merged buffer.

output: a single merged tensor for all devices.
output_sizes: a single tensor for 1st dim of outputs.
handle: Handle of a NCCL communicator.
input: a single merged tensor for all devices.
input_sizes: a tensor for 1st dim of inputs for all devices.
)doc");

#if GOOGLE_CUDA
class AlltoallvWithNcclCommOp : public NcclCommAsyncOp {
 public:
  explicit AlltoallvWithNcclCommOp(OpKernelConstruction* ctx)
      : NcclCommAsyncOp(ctx) {
    PartialTensorShape common_shape;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("common_shape", &common_shape));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("wire_dtype_for_float", &wire_dtype_for_float_));
    PartialTensorShape({1})
        .Concatenate(common_shape)
        .AsTensorShape(&output_shape_);
    common_shape_size_ = 1;
    for (int64 dim = 1; dim < output_shape_.dims(); ++dim) {
      common_shape_size_ *= output_shape_.dim_size(dim);
    }
  }

  virtual void ComputeAsyncWithComm(NcclComm* comm, OpKernelContext* ctx,
                                    DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);
    const Tensor* input_sizes;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input_sizes", &input_sizes), done);

    Tensor* sizes = new Tensor();
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT64, TensorShape({comm->size() * comm->size()}),
                           sizes, ctx->output_alloc_attr(0)),
        done);

    Tensor* host_sizes = new Tensor();
    AllocatorAttributes host_sizes_alloc_attrs;
    host_sizes_alloc_attrs.set_on_host(true);
    host_sizes_alloc_attrs.set_gpu_compatible(true);
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT64, TensorShape({comm->size() * comm->size()}),
                           host_sizes, host_sizes_alloc_attrs),
        done);

    Tensor* host_output_sizes = new Tensor();
    AllocatorAttributes host_output_sizes_alloc_attrs;
    host_output_sizes_alloc_attrs.set_on_host(true);
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT64, TensorShape({comm->size()}),
                           host_output_sizes, host_output_sizes_alloc_attrs),
        done);

    Tensor* input_in_half = new Tensor();
    Tensor* output_in_half = new Tensor();

    auto done_ = [host_sizes, host_output_sizes, input_in_half, output_in_half,
                  done]() {
      delete input_in_half;
      delete output_in_half;
      delete host_sizes;
      delete host_output_sizes;
      done();
    };
    comm->RunAsync(
        [input, input_sizes, sizes, host_sizes, host_output_sizes,
         input_in_half, output_in_half, this, comm, ctx, done_]() {
          // Collect sizes of all inputs across devices.
          VLOG(1) << comm->DebugString() << " [" << name() << "] [Allgather]";
          OP_REQUIRES_OK_ASYNC(ctx, comm->Allgather(sizes, *input_sizes),
                               done_);
          auto allgather_ready = comm->ThenRecordEvent();

          ThenWaitFor(ctx, allgather_ready);
          ThenCopyToHost(ctx, host_sizes, *sizes);
          BlockHostUntilDone(ctx);
          delete sizes;

          // Allocate outputs.
          int64 total_output_size = 0;
          for (int64 i = 0; i < comm->size(); ++i) {
            int64 output_size =
                host_sizes->flat<int64>()(comm->size() * i + comm->rank());
            total_output_size += output_size;
            host_output_sizes->flat<int64>()(i) = output_size;
          }

          TensorShape output_sizes_shape({comm->size()});
          Tensor* output_sizes;
          OP_REQUIRES_OK_ASYNC(
              ctx, ctx->allocate_output(1, output_sizes_shape, &output_sizes),
              done_);

          ThenCopyToDevice(ctx, output_sizes, *host_output_sizes);

          TensorShape output_shape(output_shape_);
          output_shape.set_dim(0, total_output_size);
          Tensor* output;
          OP_REQUIRES_OK_ASYNC(
              ctx, ctx->allocate_output(0, output_shape, &output), done_);
          auto output_ready = ThenRecordEvent(ctx);

          // Exchange data.
          for (int64 i = 0; i < comm->size() * comm->size(); ++i) {
            host_sizes->flat<int64>()(i) *= common_shape_size_;
          }
          comm->ThenWaitFor(output_ready);
          if (input->dtype() != DT_FLOAT) {
            VLOG(1) << comm->DebugString() << " [" << name() << "] [Alltoallv]";
            OP_REQUIRES_OK_ASYNC(
                ctx, comm->Alltoallv(output, *input, *host_sizes), done_);
            return;
          }
          switch (wire_dtype_for_float_) {
            case DT_HALF: {
              OP_REQUIRES_OK_ASYNC(
                  ctx,
                  ctx->allocate_temp(DT_HALF, input->shape(), input_in_half),
                  done_);
              OP_REQUIRES_OK_ASYNC(
                  ctx,
                  ctx->allocate_temp(DT_HALF, output->shape(), output_in_half),
                  done_);
              comm->ThenWaitFor(ThenRecordEvent(ctx));
              VLOG(1) << comm->DebugString() << " [" << name() << "] [CastIn]";
              functor::CastToFloat16<float> float2half_functor_;
              float2half_functor_(ctx, input, input_in_half, comm->stream());
              VLOG(1) << comm->DebugString() << " [" << name()
                      << "] [Alltoallv]";
              OP_REQUIRES_OK_ASYNC(
                  ctx,
                  comm->Alltoallv(output_in_half, *input_in_half, *host_sizes),
                  done_);
              VLOG(1) << comm->DebugString() << " [" << name() << "] [CastOut]";
              functor::CastFromFloat16<float> half2float_functor_;
              half2float_functor_(ctx,
                                  const_cast<const Tensor*>(output_in_half),
                                  output, comm->stream());
              break;
            }
            default:
              VLOG(1) << comm->DebugString() << " [" << name()
                      << "] [Alltoallv]";
              OP_REQUIRES_OK_ASYNC(
                  ctx, comm->Alltoallv(output, *input, *host_sizes), done_);
          }
        },
        ctx, done_);
  }

 private:
  TensorShape output_shape_;
  int64 common_shape_size_;
  DataType wire_dtype_for_float_;
};

#define REGISTER_KERNEL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(Name("AlltoallvWithNcclComm")   \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          AlltoallvWithNcclCommOp);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

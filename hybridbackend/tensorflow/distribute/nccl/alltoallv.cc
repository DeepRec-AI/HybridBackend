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

#include "hybridbackend/tensorflow/distribute/common/cast/functors.h"
#include "hybridbackend/tensorflow/distribute/nccl/comm.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

REGISTER_OP("HbNcclAlltoallv")
    .Output("output: dtype")
    .Output("output_sizes: int32")
    .Input("handle: resource")
    .Input("input: dtype")
    .Input("input_sizes: int32")
    .Attr("common_shape: shape = {}")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("wire_dtype: {" TF_OP_NCCL_WIRE_DTYPE_LIST "}")
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
      c->set_output(1, c->input(2));
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

namespace functor {
template <typename DTYPE, typename WIRE_DTYPE>
struct NcclAlltoallvCall {
  Status operator()(const Tensor& input, const Tensor& host_sizes,
                    const int64 common_size, Tensor* output, Tensor* comm_input,
                    Tensor* comm_output, OpKernelContext* ctx, NcclComm* comm,
                    NcclCommAsyncOp* comm_op) {
    comm->ThenWaitFor(comm_op->ThenRecordEvent(ctx));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name()
            << "] [Alltoallv]";
    TF_RETURN_IF_ERROR(comm->Alltoallv(input, host_sizes, common_size, output));
    return Status::OK();
  }
};

template <>
struct NcclAlltoallvCall<float, Eigen::half> {
  Status operator()(const Tensor& input, const Tensor& host_sizes,
                    const int64 common_size, Tensor* output, Tensor* comm_input,
                    Tensor* comm_output, OpKernelContext* ctx, NcclComm* comm,
                    NcclCommAsyncOp* comm_op) {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, input.shape(), comm_input));
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DT_HALF, output->shape(), comm_output));
    comm->ThenWaitFor(comm_op->ThenRecordEvent(ctx));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name() << "] [CastIn]";
    functor::Cast<float, Eigen::half>()(input, comm_input, ctx, comm->stream());
    VLOG(1) << comm->DebugString() << " [" << comm_op->name()
            << "] [Alltoallv]";
    TF_RETURN_IF_ERROR(
        comm->Alltoallv(*comm_input, host_sizes, common_size, comm_output));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name() << "] [CastOut]";
    functor::Cast<Eigen::half, float>()(*output, comm_output, ctx,
                                        comm->stream());
    return Status::OK();
  }
};

}  // namespace functor

template <typename DTYPE, typename WIRE_DTYPE>
class NcclAlltoallvOp : public NcclCommAsyncOp {
 public:
  explicit NcclAlltoallvOp(OpKernelConstruction* ctx) : NcclCommAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("common_shape", &common_shape_));
    common_size_ = 1;
    for (int64 dim = 0; dim < common_shape_.dims(); ++dim) {
      common_size_ *= common_shape_.dim_size(dim);
    }
  }

  virtual void ComputeAsyncWithComm(NcclComm* comm, OpKernelContext* ctx,
                                    DoneCallback done) override {
    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_on_host(true);
    host_alloc_attrs.set_gpu_compatible(true);

    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);
    const Tensor* input_sizes;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input_sizes", &input_sizes), done);

    Tensor* sizes = new Tensor();
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT32, TensorShape({comm->size() * comm->size()}),
                           sizes, ctx->output_alloc_attr(0)),
        done);

    Tensor* host_sizes = new Tensor();
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT32, TensorShape({comm->size() * comm->size()}),
                           host_sizes, host_alloc_attrs),
        done);

    Tensor* host_output_sizes = new Tensor();
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT32, TensorShape({comm->size()}),
                           host_output_sizes, host_alloc_attrs),
        done);

    Tensor* comm_input = new Tensor();
    Tensor* comm_output = new Tensor();
    auto done_ = [host_sizes, host_output_sizes, comm_input, comm_output,
                  done]() {
      delete host_sizes;
      delete host_output_sizes;
      delete comm_input;
      delete comm_output;
      done();
    };
    comm->RunAsync(
        "NcclAlltoallv", ctx, done_,
        [input_sizes, sizes, host_output_sizes, input, host_sizes, comm_input,
         comm_output, ctx, comm, this, done_]() {
          // Collect sizes of all inputs across devices.
          VLOG(1) << comm->DebugString() << " [" << name() << "] [Allgather]";
          OP_REQUIRES_OK_ASYNC(ctx, comm->Allgather(*input_sizes, sizes),
                               done_);
          ThenWaitFor(ctx, comm->ThenRecordEvent());
          ThenCopyToHost(ctx, host_sizes, *sizes);
          BlockHostUntilDone(ctx);
          delete sizes;

          // Allocate outputs.
          int32 total_output_size = 0;
          for (int32 i = 0; i < comm->size(); ++i) {
            int32 output_size =
                host_sizes->flat<int32>()(comm->size() * i + comm->rank());
            total_output_size += output_size;
            host_output_sizes->flat<int32>()(i) = output_size;
          }

          TensorShape output_sizes_shape({comm->size()});
          Tensor* output_sizes;
          OP_REQUIRES_OK_ASYNC(
              ctx, ctx->allocate_output(1, output_sizes_shape, &output_sizes),
              done_);
          ThenCopyToDevice(ctx, output_sizes, *host_output_sizes);

          TensorShape output_shape;
          PartialTensorShape({total_output_size})
              .Concatenate(common_shape_)
              .AsTensorShape(&output_shape);
          Tensor* output;
          OP_REQUIRES_OK_ASYNC(
              ctx, ctx->allocate_output(0, output_shape, &output), done_);

          // Cast and communicate.
          auto call = functor::NcclAlltoallvCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(ctx,
                               call(*input, *host_sizes, common_size_, output,
                                    comm_input, comm_output, ctx, comm, this),
                               done_);
        });
  }

 private:
  PartialTensorShape common_shape_;
  int64 common_size_;
};

#define REGISTER_KERNEL(DTYPE, WIRE_DTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAlltoallv")                        \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<DTYPE>("dtype")            \
                              .TypeConstraint<WIRE_DTYPE>("wire_dtype"), \
                          NcclAlltoallvOp<DTYPE, WIRE_DTYPE>);
TF_CALL_NCCL_CAST_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

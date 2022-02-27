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

REGISTER_OP("AlltoallwWithNcclComm")
    .Output("outputs: P * T")
    .Input("handle: resource")
    .Input("inputs: P * T")
    .Attr("common_shape: shape = {}")
    .Attr("wire_dtype_for_float: {half, float}")
    .Attr("P: int >= 1 = 1")
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
      for (int64 dim = 0; dim < c->num_outputs(); ++dim) {
        c->set_output(dim, shape);
      }
      return Status::OK();
    })
    .Doc(R"doc(
AllToAllw using a NCCL communicator.

outputs: Rotated tensors for each device.
handle: Handle of a NCCL communicator.
inputs: Tensors to rotate for each device.
)doc");

#if GOOGLE_CUDA
class AlltoallwWithNcclCommOp : public NcclCommAsyncOp {
 public:
  explicit AlltoallwWithNcclCommOp(OpKernelConstruction* ctx)
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
    OpInputList inputs;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("inputs", &inputs), done);
    std::vector<Tensor>* inputs_vec = new std::vector<Tensor>();
    for (int i = 0; i < comm->size(); ++i) {
      inputs_vec->push_back(inputs[i]);
    }

    ctx->set_output(comm->rank(), inputs[comm->rank()]);

    Tensor* host_sizes = new Tensor();
    AllocatorAttributes host_sizes_alloc_attrs;
    host_sizes_alloc_attrs.set_on_host(true);
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT64, TensorShape({comm->size() * comm->size()}),
                           host_sizes, host_sizes_alloc_attrs),
        done);
    std::vector<Tensor>* inputs_vec_in_half = new std::vector<Tensor>();
    std::vector<Tensor*>* outputs_vec_in_half = new std::vector<Tensor*>();

    auto done_ = [inputs_vec, inputs_vec_in_half, outputs_vec_in_half, done]() {
      for (auto e : (*outputs_vec_in_half)) {
        delete e;
      }
      delete inputs_vec_in_half;
      delete outputs_vec_in_half;
      delete inputs_vec;
      done();
    };
    comm->RunAsync(
        [inputs_vec, inputs_vec_in_half, outputs_vec_in_half, host_sizes, this,
         comm, ctx, done_]() {
          // Allocate scratch tensors.
          Tensor* input_sizes = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64, TensorShape({comm->size()}),
                                 input_sizes, ctx->output_alloc_attr(0)),
              done_);

          Tensor host_input_sizes;
          AllocatorAttributes host_input_sizes_alloc_attrs;
          host_input_sizes_alloc_attrs.set_on_host(true);
          host_input_sizes_alloc_attrs.set_gpu_compatible(true);
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64, TensorShape({comm->size()}),
                                 &host_input_sizes,
                                 host_input_sizes_alloc_attrs),
              done_);
          for (int i = 0; i < comm->size(); ++i) {
            host_input_sizes.flat<int64>()(i) = inputs_vec->at(i).NumElements();
          }

          ThenCopyToDevice(ctx, input_sizes, host_input_sizes);

          Tensor* sizes = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64,
                                 TensorShape({comm->size() * comm->size()}),
                                 sizes, ctx->output_alloc_attr(0)),
              done_);
          auto sizes_ready = ThenRecordEvent(ctx);

          comm->ThenWaitFor(sizes_ready);
          comm->BlockHostUntilDone();

          // Collect sizes of all inputs across devices.
          VLOG(1) << comm->DebugString() << " [" << name() << "] [Allgather]";
          OP_REQUIRES_OK_ASYNC(ctx, comm->Allgather(sizes, *input_sizes),
                               done_);
          auto allgather_ready = comm->ThenRecordEvent();

          ThenWaitFor(ctx, allgather_ready);
          ThenCopyToHost(ctx, host_sizes, *sizes);
          BlockHostUntilDone(ctx);
          delete input_sizes;
          delete sizes;

          std::vector<int64> output_sizes_vec;
          for (int64 i = 0; i < host_sizes->NumElements(); ++i) {
            const int64 output_elements = host_sizes->flat<int64>()(i);
            OP_REQUIRES_ASYNC(
                ctx, output_elements % common_shape_size_ == 0,
                errors::InvalidArgument(
                    "common_shape size ", common_shape_size_,
                    " is not compatible with input ", i, ": ", output_elements),
                done_);
            const int64 output_size = output_elements / common_shape_size_;
            output_sizes_vec.push_back(output_size);
          }
          delete host_sizes;

          // Allocate outputs for non-symmetric inputs.
          for (int i = 0; i < comm->size(); ++i) {
            if (i == comm->rank()) {
              continue;
            }
            TensorShape output_shape(output_shape_);
            output_shape.set_dim(
                0, output_sizes_vec[comm->size() * i + comm->rank()]);
            Tensor* output;
            OP_REQUIRES_OK_ASYNC(
                ctx, ctx->allocate_output(i, output_shape, &output), done_);
          }
          OpOutputList outputs;
          OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("outputs", &outputs),
                               done_);
          auto outputs_ready = ThenRecordEvent(ctx);

          // Exchange partial inputs.
          std::vector<Tensor*> outputs_vec;
          for (int i = 0; i < comm->size(); ++i) {
            outputs_vec.push_back(outputs[i]);
          }
          comm->ThenWaitFor(outputs_ready);
          if (inputs_vec->at(0).dtype() != DT_FLOAT) {
            VLOG(1) << comm->DebugString() << " [" << name() << "] [Alltoallw]";
            OP_REQUIRES_OK_ASYNC(
                ctx, comm->Alltoallw(&outputs_vec, *inputs_vec), done_);
            return;
          }
          switch (wire_dtype_for_float_) {
            case DT_HALF: {
              for (auto e : *inputs_vec) {
                Tensor elem_in_half;
                OP_REQUIRES_OK_ASYNC(
                    ctx, ctx->allocate_temp(DT_HALF, e.shape(), &elem_in_half),
                    done_);
                inputs_vec_in_half->push_back(elem_in_half);
              }
              for (auto e : outputs_vec) {
                Tensor* elem_in_half = new Tensor();
                OP_REQUIRES_OK_ASYNC(
                    ctx, ctx->allocate_temp(DT_HALF, e->shape(), elem_in_half),
                    done_);
                outputs_vec_in_half->push_back(elem_in_half);
              }
              comm->ThenWaitFor(ThenRecordEvent(ctx));
              VLOG(1) << comm->DebugString() << " [" << name() << "] [CastIn]";
              functor::CastToFloat16<float> float2half_functor_;
              for (int i = 0; i < inputs_vec->size(); i++) {
                float2half_functor_(ctx, &((*inputs_vec)[i]),
                                    &((*inputs_vec_in_half)[i]),
                                    comm->stream());
              }
              VLOG(1) << comm->DebugString() << " [" << name()
                      << "] [Alltoallw]";
              OP_REQUIRES_OK_ASYNC(
                  ctx,
                  comm->Alltoallw(outputs_vec_in_half, *inputs_vec_in_half),
                  done_);
              VLOG(1) << comm->DebugString() << " [" << name() << "] [CastOut]";
              functor::CastFromFloat16<float> half2float_functor_;
              for (int i = 0; i < outputs_vec_in_half->size(); i++) {
                half2float_functor_(
                    ctx, const_cast<const Tensor*>((*outputs_vec_in_half)[i]),
                    outputs_vec[i], comm->stream());
              }
              break;
            }
            default:
              VLOG(1) << comm->DebugString() << " [" << name()
                      << "] [Alltoallw]";
              OP_REQUIRES_OK_ASYNC(
                  ctx, comm->Alltoallw(&outputs_vec, *inputs_vec), done_);
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
  REGISTER_KERNEL_BUILDER(Name("AlltoallwWithNcclComm")   \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          AlltoallwWithNcclCommOp);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

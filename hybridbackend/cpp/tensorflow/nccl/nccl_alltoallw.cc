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
    .Output("outputs: num_shards * dtype")
    .Input("handle: resource")
    .Input("inputs: num_shards * dtype")
    .Attr("common_shape: shape = {}")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("wire_dtype: {" TF_OP_NCCL_WIRE_DTYPE_LIST "}")
    .Attr("num_shards: int >= 1 = 1")
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

namespace functor {
template <typename DTYPE, typename WIRE_DTYPE>
struct AlltoallwCall {
  Status operator()(const std::vector<Tensor>& inputs_vec,
                    std::vector<Tensor*>* outputs_vec,
                    std::vector<Tensor*>* comm_input_vec,
                    std::vector<Tensor*>* comm_output_vec, OpKernelContext* ctx,
                    NcclComm* comm, NcclCommAsyncOp* comm_op) {
    comm->ThenWaitFor(comm_op->ThenRecordEvent(ctx));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name()
            << "] [Alltoallw]";
    TF_RETURN_IF_ERROR(comm->Alltoallw(inputs_vec, outputs_vec));
    return Status::OK();
  }
};

template <>
struct AlltoallwCall<float, Eigen::half> {
  Status operator()(const std::vector<Tensor>& inputs_vec,
                    std::vector<Tensor*>* outputs_vec,
                    std::vector<Tensor*>* comm_input_vec,
                    std::vector<Tensor*>* comm_output_vec, OpKernelContext* ctx,
                    NcclComm* comm, NcclCommAsyncOp* comm_op) {
    for (int i = 0; i < comm->size(); ++i) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, inputs_vec[i].shape(),
                                            comm_input_vec->at(i)));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DT_HALF, outputs_vec->at(i)->shape(), comm_output_vec->at(i)));
    }
    comm->ThenWaitFor(comm_op->ThenRecordEvent(ctx));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name() << "] [CastIn]";
    for (int i = 0; i < inputs_vec.size(); i++) {
      functor::Cast<float, Eigen::half>()(inputs_vec[i], comm_input_vec->at(i),
                                          ctx, comm->stream());
    }
    VLOG(1) << comm->DebugString() << " [" << comm_op->name()
            << "] [Alltoallw]";
    TF_RETURN_IF_ERROR(comm->Alltoallw(*comm_input_vec, comm_output_vec));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name() << "] [CastOut]";
    for (int i = 0; i < comm_output_vec->size(); i++) {
      functor::Cast<Eigen::half, float>()(
          *(comm_output_vec->at(i)), outputs_vec->at(i), ctx, comm->stream());
    }
    return Status::OK();
  }
};

}  // namespace functor

template <typename DTYPE, typename WIRE_DTYPE>
class AlltoallwWithNcclCommOp : public NcclCommAsyncOp {
 public:
  explicit AlltoallwWithNcclCommOp(OpKernelConstruction* ctx)
      : NcclCommAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("common_shape", &common_shape_));
    common_size_ = 1;
    for (int64 dim = 0; dim < common_shape_.dims(); ++dim) {
      common_size_ *= common_shape_.dim_size(dim);
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

    std::vector<Tensor*>* comm_input_vec = new std::vector<Tensor*>();
    std::vector<Tensor*>* comm_output_vec = new std::vector<Tensor*>();
    for (int i = 0; i < comm->size(); ++i) {
      comm_input_vec->push_back(new Tensor());
      comm_output_vec->push_back(new Tensor());
    }
    auto done_ = [inputs_vec, comm_input_vec, comm_output_vec, done]() {
      for (auto t : (*comm_input_vec)) {
        delete t;
      }
      delete comm_input_vec;
      for (auto t : (*comm_output_vec)) {
        delete t;
      }
      delete comm_output_vec;
      delete inputs_vec;
      done();
    };

    comm->RunAsync(
        [inputs_vec, comm_input_vec, comm_output_vec, this, comm, ctx,
         done_]() {
          AllocatorAttributes host_alloc_attrs;
          host_alloc_attrs.set_on_host(true);
          host_alloc_attrs.set_gpu_compatible(true);

          // Allocate scratch tensors.
          Tensor* input_sizes = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64, TensorShape({comm->size()}),
                                 input_sizes, ctx->output_alloc_attr(0)),
              done_);

          Tensor host_input_sizes;
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64, TensorShape({comm->size()}),
                                 &host_input_sizes, host_alloc_attrs),
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

          Tensor* host_sizes = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64,
                                 TensorShape({comm->size() * comm->size()}),
                                 host_sizes, host_alloc_attrs),
              done_);
          auto sizes_ready = ThenRecordEvent(ctx);

          comm->ThenWaitFor(sizes_ready);
          comm->BlockHostUntilDone();

          // Collect sizes of all inputs across devices.
          VLOG(1) << comm->DebugString() << " [" << name() << "] [Allgather]";
          OP_REQUIRES_OK_ASYNC(ctx, comm->Allgather(*input_sizes, sizes),
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
                ctx, output_elements % common_size_ == 0,
                errors::InvalidArgument("common_shape size ", common_size_,
                                        " is not compatible with input ", i,
                                        ": ", output_elements),
                done_);
            const int64 output_size = output_elements / common_size_;
            output_sizes_vec.push_back(output_size);
          }
          delete host_sizes;

          // Allocate outputs for non-symmetric inputs.
          for (int i = 0; i < comm->size(); ++i) {
            if (i == comm->rank()) {
              continue;
            }
            TensorShape output_shape;
            PartialTensorShape(
                {output_sizes_vec[comm->size() * i + comm->rank()]})
                .Concatenate(common_shape_)
                .AsTensorShape(&output_shape);
            Tensor* output;
            OP_REQUIRES_OK_ASYNC(
                ctx, ctx->allocate_output(i, output_shape, &output), done_);
          }
          OpOutputList outputs;
          OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("outputs", &outputs),
                               done_);

          // Exchange partial inputs.
          std::vector<Tensor*> outputs_vec;
          for (int i = 0; i < comm->size(); ++i) {
            outputs_vec.push_back(outputs[i]);
          }

          // Cast and communicate.
          auto call = functor::AlltoallwCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(ctx,
                               call(*inputs_vec, &outputs_vec, comm_input_vec,
                                    comm_output_vec, ctx, comm, this),
                               done_);
        },
        ctx, done_);
  }

 private:
  PartialTensorShape common_shape_;
  int64 common_size_;
};

#define REGISTER_KERNEL(DTYPE, WIRE_DTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("AlltoallwWithNcclComm")                  \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<DTYPE>("dtype")            \
                              .TypeConstraint<WIRE_DTYPE>("wire_dtype"), \
                          AlltoallwWithNcclCommOp<DTYPE, WIRE_DTYPE>);
TF_CALL_NCCL_CAST_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

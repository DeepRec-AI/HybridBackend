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

#include <absl/strings/str_cat.h>

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <vector>

#include "hybridbackend/tensorflow/distribute/common/cast/functors.h"
#include "hybridbackend/tensorflow/distribute/nccl/comm.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

REGISTER_OP("HbNcclAlltoallwN")
    .Output("outputs: num_shards * dtype")
    .Input("handle: resource")
    .Input("inputs: num_shards * dtype")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("wire_dtype: {" TF_OP_NCCL_WIRE_DTYPE_LIST "}")
    .Attr("num_shards: int >= 1 = 1")
    .Attr("num_columns: int >= 1 = 1")
    .Attr("common_shapes: list(shape)")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 num_columns;
      TF_RETURN_IF_ERROR(c->GetAttr("num_columns", &num_columns));
      int64 comm_size = c->num_outputs() / num_columns;
      std::vector<PartialTensorShape> common_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("common_shapes", &common_shapes));
      for (int64 i = 0; i < num_columns; ++i) {
        shape_inference::ShapeHandle shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(common_shapes[i], &shape));
        TF_RETURN_IF_ERROR(c->Concatenate(
            c->Vector(shape_inference::InferenceContext::kUnknownDim), shape,
            &shape));
        for (int64 dim = i * comm_size; dim < (i + 1) * comm_size; ++dim) {
          c->set_output(dim, shape);
        }
      }
      return Status::OK();
    })
    .Doc(R"doc(
Grouped AllToAllw using a NCCL communicator.

outputs: Rotated tensors for each device.
handle: Handle of a NCCL communicator.
inputs: Tensors to rotate for each device.
)doc");

#if GOOGLE_CUDA

namespace functor {
template <typename DTYPE, typename WIRE_DTYPE>
struct NcclAlltoallwNCall {
  Status operator()(const std::vector<Tensor>& inputs_vec,
                    const int64 num_columns, std::vector<Tensor*>* outputs_vec,
                    std::vector<Tensor*>* comm_input_vec,
                    std::vector<Tensor*>* comm_output_vec, OpKernelContext* ctx,
                    NcclComm* comm, NcclCommAsyncOp* comm_op) {
    comm->ThenWaitFor(comm_op->ThenRecordEvent(ctx));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name()
            << "] [AlltoallwN]";
    TF_RETURN_IF_ERROR(comm->AlltoallwN(inputs_vec, num_columns, outputs_vec));
    return Status::OK();
  }
};

template <>
struct NcclAlltoallwNCall<float, Eigen::half> {
  Status operator()(const std::vector<Tensor>& inputs_vec,
                    const int64 num_columns, std::vector<Tensor*>* outputs_vec,
                    std::vector<Tensor*>* comm_input_vec,
                    std::vector<Tensor*>* comm_output_vec, OpKernelContext* ctx,
                    NcclComm* comm, NcclCommAsyncOp* comm_op) {
    for (int i = 0; i < inputs_vec.size(); ++i) {
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
            << "] [AlltoallwN]";
    TF_RETURN_IF_ERROR(
        comm->AlltoallwN(*comm_input_vec, num_columns, comm_output_vec));
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
class NcclAlltoallwNOp : public NcclCommAsyncOp {
 public:
  explicit NcclAlltoallwNOp(OpKernelConstruction* ctx) : NcclCommAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_columns", &num_columns_));
    std::vector<PartialTensorShape> common_shapes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("common_shapes", &common_shapes));

    for (int64 c = 0; c < num_columns_; ++c) {
      TensorShape output_shape;
      PartialTensorShape({1})
          .Concatenate(common_shapes[c])
          .AsTensorShape(&output_shape);
      int64 common_shape_size = 1;
      for (int64 dim = 1; dim < output_shape.dims(); ++dim) {
        common_shape_size *= output_shape.dim_size(dim);
      }
      output_shapes_.push_back(std::move(output_shape));
      common_shape_sizes_.push_back(common_shape_size);
    }
  }

  virtual void ComputeAsyncWithComm(NcclComm* comm, OpKernelContext* ctx,
                                    DoneCallback done) override {
    // Get num_columns * comm_size inputs.
    // e.g. [tensor0_for_gpu0, tensor0_for_gpu1,
    //       tensor1_for_gpu0, tensor1_for_gpu1,
    //       tensor2_for_gpu0, tensor2_for_gpu1]

    AllocatorAttributes local_alloc_attrs;
    local_alloc_attrs.set_on_host(true);
    local_alloc_attrs.set_gpu_compatible(true);

    OpInputList inputs;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("inputs", &inputs), done);

    // Collect sizes of all inputs across devices.
    const int64 inputs_size = inputs.size();
    std::vector<Tensor>* inputs_vec = new std::vector<Tensor>();
    for (int i = 0; i < inputs_size; ++i) {
      inputs_vec->push_back(inputs[i]);
    }
    std::vector<Tensor*>* comm_input_vec = new std::vector<Tensor*>();
    for (int i = 0; i < inputs_size; ++i) {
      comm_input_vec->push_back(new Tensor());
    }
    std::vector<Tensor*>* comm_output_vec = new std::vector<Tensor*>();
    for (int i = 0; i < inputs_size; ++i) {
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
        "NcclAlltoallwN", ctx, done_,
        [inputs_vec, comm_input_vec, comm_output_vec, this, inputs_size, comm,
         ctx, done_]() {
          AllocatorAttributes local_alloc_attrs;
          local_alloc_attrs.set_on_host(true);
          local_alloc_attrs.set_gpu_compatible(true);

          // Allocate scratch tensors.
          Tensor* local_sizes = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64, TensorShape({inputs_size}),
                                 local_sizes, ctx->output_alloc_attr(0)),
              done_);

          Tensor host_local_sizes;
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64, TensorShape({inputs_size}),
                                 &host_local_sizes, local_alloc_attrs),
              done_);
          for (int i = 0; i < inputs_size; ++i) {
            host_local_sizes.flat<int64>()(i) = inputs_vec->at(i).NumElements();
          }
          ThenCopyToDevice(ctx, local_sizes, host_local_sizes);

          Tensor* all_sizes = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64,
                                 TensorShape({inputs_size * comm->size()}),
                                 all_sizes, ctx->output_alloc_attr(0)),
              done_);

          Tensor* host_all_sizes = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64,
                                 TensorShape({inputs_size * comm->size()}),
                                 host_all_sizes, local_alloc_attrs),
              done_);

          comm->ThenWaitFor(ThenRecordEvent(ctx));
          comm->BlockHostUntilDone();

          // Collect sizes of all inputs across devices.
          VLOG(1) << comm->DebugString() << " [" << name() << "] [Allgather]";
          OP_REQUIRES_OK_ASYNC(ctx, comm->Allgather(*local_sizes, all_sizes),
                               done_);
          ThenWaitFor(ctx, comm->ThenRecordEvent());
          ThenCopyToHost(ctx, host_all_sizes, *all_sizes);
          BlockHostUntilDone(ctx);
          delete local_sizes;
          delete all_sizes;

          std::vector<int64> output_sizes_vec;
          for (int64 i = 0; i < host_all_sizes->NumElements(); ++i) {
            int64 to_idx = (i % inputs_size) % comm->size();
            int64 c = (i % inputs_size) / comm->size();
            int64 output_size = host_all_sizes->flat<int64>()(i);
            int64 common_shape_size = common_shape_sizes_[c];
            OP_REQUIRES_ASYNC(ctx, output_size % common_shape_size == 0,
                              errors::InvalidArgument(absl::StrCat(
                                  "[GPU", comm->rank(), "] Tensor ", c,
                                  ": common_shape (size=", common_shape_size,
                                  ") is not compatible with input ", to_idx,
                                  " (size=", output_size, ")")),
                              done_);
            output_sizes_vec.push_back(output_size / common_shape_size);
          }
          delete host_all_sizes;

          // Redirect symmetric inputs to outputs.
          for (int64 c = 0; c < num_columns_; ++c) {
            int64 fixed_idx = c * comm->size() + comm->rank();
            ctx->set_output(fixed_idx, inputs_vec->at(fixed_idx));
          }

          // Allocate outputs for non-symmetric inputs.
          for (int gpu_idx = 0; gpu_idx < comm->size(); ++gpu_idx) {
            if (gpu_idx == comm->rank()) {
              continue;
            }
            for (int64 c = 0; c < num_columns_; ++c) {
              TensorShape output_shape(output_shapes_[c]);
              output_shape.set_dim(
                  0, output_sizes_vec[comm->size() * num_columns_ * gpu_idx +
                                      comm->size() * c + comm->rank()]);
              Tensor* output;
              OP_REQUIRES_OK_ASYNC(
                  ctx,
                  ctx->allocate_output(comm->size() * c + gpu_idx, output_shape,
                                       &output),
                  done_);
            }
          }

          OpOutputList outputs;
          OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("outputs", &outputs),
                               done_);

          // Exchange partial inputs.
          std::vector<Tensor*> outputs_vec;
          for (int i = 0; i < outputs.size(); ++i) {
            outputs_vec.push_back(outputs[i]);
          }
          // Cast and communicate.
          auto call = functor::NcclAlltoallwNCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              call(*inputs_vec, num_columns_, &outputs_vec, comm_input_vec,
                   comm_output_vec, ctx, comm, this),
              done_);
        });
  }

 private:
  int64 num_columns_;
  std::vector<TensorShape> output_shapes_;
  std::vector<int64> common_shape_sizes_;
};

#define REGISTER_KERNEL(DTYPE, WIRE_DTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAlltoallwN")                       \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<DTYPE>("dtype")            \
                              .TypeConstraint<WIRE_DTYPE>("wire_dtype"), \
                          NcclAlltoallwNOp<DTYPE, WIRE_DTYPE>);
TF_CALL_NCCL_CAST_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

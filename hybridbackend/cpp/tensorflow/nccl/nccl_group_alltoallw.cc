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

REGISTER_OP("GroupAlltoallwWithNcclComm")
    .Output("outputs: P * T")
    .Input("handle: resource")
    .Input("inputs: P * T")
    .Attr("group_size: int >= 1 = 1")
    .Attr("common_shapes: list(shape)")
    .Attr("rank: int >= 0 = 0")
    .Attr("wire_dtype_for_float: {half, float}")
    .Attr("P: int >= 1 = 1")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 group_size;
      TF_RETURN_IF_ERROR(c->GetAttr("group_size", &group_size));
      int64 num_gpus = c->num_outputs() / group_size;
      std::vector<PartialTensorShape> common_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("common_shapes", &common_shapes));
      for (int64 tensor_idx = 0; tensor_idx < group_size; ++tensor_idx) {
        shape_inference::ShapeHandle shape;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
            common_shapes[tensor_idx], &shape));
        TF_RETURN_IF_ERROR(c->Concatenate(
            c->Vector(shape_inference::InferenceContext::kUnknownDim), shape,
            &shape));
        for (int64 dim = tensor_idx * num_gpus;
             dim < (tensor_idx + 1) * num_gpus; ++dim) {
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
rank: Index of current device in the communicator.
)doc");

#if GOOGLE_CUDA
class GroupAlltoallwWithNcclCommOp : public NcclCommAsyncOp {
 public:
  explicit GroupAlltoallwWithNcclCommOp(OpKernelConstruction* ctx)
      : NcclCommAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("group_size", &group_size_));
    std::vector<PartialTensorShape> common_shapes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("common_shapes", &common_shapes));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("wire_dtype_for_float", &wire_dtype_for_float_));

    for (int64 tensor_idx = 0; tensor_idx < group_size_; ++tensor_idx) {
      TensorShape output_shape;
      PartialTensorShape({1})
          .Concatenate(common_shapes[tensor_idx])
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
    // Get group_size * num_gpus inputs.
    // e.g. [tensor0_for_gpu0, tensor0_for_gpu1,
    //       tensor1_for_gpu0, tensor1_for_gpu1,
    //       tensor2_for_gpu0, tensor2_for_gpu1]

    OpInputList inputs;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("inputs", &inputs), done);

    // Collect sizes of all inputs across devices.
    const int64 inputs_size = inputs.size();
    std::vector<Tensor>* inputs_vec = new std::vector<Tensor>();
    for (int i = 0; i < inputs_size; ++i) {
      inputs_vec->push_back(inputs[i]);
    }

    AllocatorAttributes all_sizes_alloc_attrs;
    all_sizes_alloc_attrs.set_on_host(true);
    Tensor* all_sizes_on_host = new Tensor();
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT64, TensorShape({inputs_size * comm->size()}),
                           all_sizes_on_host, all_sizes_alloc_attrs),
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
        [inputs_vec, inputs_vec_in_half, outputs_vec_in_half, all_sizes_on_host,
         this, inputs_size, comm, ctx, done_]() {
          // Allocate scratch tensors.
          Tensor* local_sizes_on_gpu = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64, TensorShape({inputs_size}),
                                 local_sizes_on_gpu, ctx->output_alloc_attr(0)),
              done_);

          Tensor local_sizes_on_host;
          AllocatorAttributes local_sizes_alloc_attrs;
          local_sizes_alloc_attrs.set_on_host(true);
          local_sizes_alloc_attrs.set_gpu_compatible(true);

          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64, TensorShape({inputs_size}),
                                 &local_sizes_on_host, local_sizes_alloc_attrs),
              done_);

          for (int i = 0; i < inputs_size; ++i) {
            local_sizes_on_host.flat<int64>()(i) =
                inputs_vec->at(i).NumElements();
          }

          ThenCopyToDevice(ctx, local_sizes_on_gpu, local_sizes_on_host);

          Tensor* all_sizes_on_gpu = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT64,
                                 TensorShape({inputs_size * comm->size()}),
                                 all_sizes_on_gpu, ctx->output_alloc_attr(0)),
              done_);
          auto sizes_ready = ThenRecordEvent(ctx);

          comm->ThenWaitFor(sizes_ready);
          comm->BlockHostUntilDone();

          // Collect sizes of all inputs across devices.
          VLOG(1) << comm->DebugString() << " [" << name() << "] [Allgather]";
          OP_REQUIRES_OK_ASYNC(
              ctx, comm->Allgather(all_sizes_on_gpu, *local_sizes_on_gpu),
              done_);
          auto allgather_ready = comm->ThenRecordEvent();

          ThenWaitFor(ctx, allgather_ready);
          ThenCopyToHost(ctx, all_sizes_on_host, *all_sizes_on_gpu);
          BlockHostUntilDone(ctx);
          delete local_sizes_on_gpu;
          delete all_sizes_on_gpu;

          std::vector<int64> output_sizes_vec;
          for (int64 i = 0; i < all_sizes_on_host->NumElements(); ++i) {
            int64 to_idx = (i % inputs_size) % comm->size();
            int64 tensor_idx = (i % inputs_size) / comm->size();
            int64 output_size = all_sizes_on_host->flat<int64>()(i);
            int64 common_shape_size = common_shape_sizes_[tensor_idx];
            OP_REQUIRES_ASYNC(ctx, output_size % common_shape_size == 0,
                              errors::InvalidArgument(strings::StrCat(
                                  "[GPU", comm->rank(), "] Tensor ", tensor_idx,
                                  ": common_shape (size=", common_shape_size,
                                  ") is not compatible with input ", to_idx,
                                  " (size=", output_size, ")")),
                              done_);
            output_sizes_vec.push_back(output_size / common_shape_size);
          }

          delete all_sizes_on_host;

          // Redirect symmetric inputs to outputs.
          for (int64 tensor_idx = 0; tensor_idx < group_size_; ++tensor_idx) {
            int64 fixed_idx = tensor_idx * comm->size() + comm->rank();
            ctx->set_output(fixed_idx, inputs_vec->at(fixed_idx));
          }

          // Allocate outputs for non-symmetric inputs.
          for (int gpu_idx = 0; gpu_idx < comm->size(); ++gpu_idx) {
            if (gpu_idx == comm->rank()) {
              continue;
            }
            for (int64 tensor_idx = 0; tensor_idx < group_size_; ++tensor_idx) {
              TensorShape output_shape(output_shapes_[tensor_idx]);
              output_shape.set_dim(
                  0,
                  output_sizes_vec[comm->size() * group_size_ * gpu_idx +
                                   comm->size() * tensor_idx + comm->rank()]);
              Tensor* output;
              OP_REQUIRES_OK_ASYNC(
                  ctx,
                  ctx->allocate_output(comm->size() * tensor_idx + gpu_idx,
                                       output_shape, &output),
                  done_);
            }
          }

          OpOutputList outputs;
          OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("outputs", &outputs),
                               done_);
          auto outputs_ready = ThenRecordEvent(ctx);

          // Exchange partial inputs.
          std::vector<Tensor*> outputs_vec;
          for (int i = 0; i < outputs.size(); ++i) {
            outputs_vec.push_back(outputs[i]);
          }
          comm->ThenWaitFor(outputs_ready);
          if (inputs_vec->at(0).dtype() != DT_FLOAT) {
            VLOG(1) << comm->DebugString() << " [" << name()
                    << "] [GroupAlltoallw]";
            OP_REQUIRES_OK_ASYNC(
                ctx,
                comm->GroupAlltoallw(&outputs_vec, *inputs_vec, group_size_),
                done_);
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
                      << "] [GroupAlltoallw]";
              OP_REQUIRES_OK_ASYNC(
                  ctx,
                  comm->GroupAlltoallw(outputs_vec_in_half, *inputs_vec_in_half,
                                       group_size_),
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
                      << "] [GroupAlltoallw]";
              OP_REQUIRES_OK_ASYNC(
                  ctx,
                  comm->GroupAlltoallw(&outputs_vec, *inputs_vec, group_size_),
                  done_);
          }
        },
        ctx, done_);
  }

 private:
  int64 group_size_;
  std::vector<TensorShape> output_shapes_;
  std::vector<int64> common_shape_sizes_;
  DataType wire_dtype_for_float_;
};

#define REGISTER_KERNEL(TYPE)                                \
  REGISTER_KERNEL_BUILDER(Name("GroupAlltoallwWithNcclComm") \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<TYPE>("T"),    \
                          GroupAlltoallwWithNcclCommOp);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

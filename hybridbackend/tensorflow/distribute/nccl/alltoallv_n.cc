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

#include "hybridbackend/tensorflow/common/host_functions.h"
#include "hybridbackend/tensorflow/distribute/common/cast/functors.h"
#include "hybridbackend/tensorflow/distribute/common/slice_sum/functors.h"
#include "hybridbackend/tensorflow/distribute/nccl/comm.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

REGISTER_OP("HbNcclAlltoallvN")
    .Output("outputs: num_columns * dtype")
    .Output("outputs_sizes: num_columns * int32")
    .Input("handle: resource")
    .Input("inputs: num_columns * dtype")
    .Input("inputs_sizes: num_columns * int32")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("wire_dtype: {" TF_OP_NCCL_WIRE_DTYPE_LIST "}")
    .Attr("num_columns: int >= 1 = 1")
    .Attr("common_shapes: list(shape)")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 num_columns;
      TF_RETURN_IF_ERROR(c->GetAttr("num_columns", &num_columns));
      std::vector<PartialTensorShape> common_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("common_shapes", &common_shapes));
      for (int64 n = 0; n < num_columns; ++n) {
        shape_inference::ShapeHandle shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(common_shapes[n], &shape));
        TF_RETURN_IF_ERROR(c->Concatenate(
            c->Vector(shape_inference::InferenceContext::kUnknownDim), shape,
            &shape));
        c->set_output(n, shape);
        c->set_output(num_columns + n, c->input(1 + num_columns + n));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Grouped AllToAllv using a NCCL communicator with merged buffer.

outputs: Merged tensors for all devices.
outputs_sizes: Tensors for 1st dim of outputs.
handle: Handle of a NCCL communicator.
inputs: Merged tensors for all devices.
inputs_sizes: Tensors for 1st dim of inputs for all devices.
)doc");

#if GOOGLE_CUDA

namespace functor {
template <typename DTYPE, typename WIRE_DTYPE>
struct NcclAlltoallvNCall {
  Status operator()(const std::vector<Tensor>& inputs, const Tensor& host_sizes,
                    const std::vector<int64>& common_sizes,
                    std::vector<Tensor*>* outputs,
                    std::vector<Tensor*>* comm_in_vec,
                    std::vector<Tensor*>* comm_out_vec, OpKernelContext* ctx,
                    NcclComm* comm, NcclCommAsyncOp* comm_op) {
    comm->ThenWaitFor(comm_op->ThenRecordEvent(ctx));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name()
            << "] [AlltoallvN]";
    TF_RETURN_IF_ERROR(
        comm->AlltoallvN(inputs, host_sizes, common_sizes, outputs));
    return Status::OK();
  }
};

template <>
struct NcclAlltoallvNCall<float, Eigen::half> {
  Status operator()(const std::vector<Tensor>& inputs, const Tensor& host_sizes,
                    const std::vector<int64>& common_sizes,
                    std::vector<Tensor*>* outputs,
                    std::vector<Tensor*>* comm_input_vec,
                    std::vector<Tensor*>* comm_output_vec, OpKernelContext* ctx,
                    NcclComm* comm, NcclCommAsyncOp* comm_op) {
    for (int i = 0; i < inputs.size(); ++i) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, inputs[i].shape(),
                                            comm_input_vec->at(i),
                                            ctx->input_alloc_attr(i)));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, outputs->at(i)->shape(),
                                            comm_output_vec->at(i),
                                            ctx->output_alloc_attr(i)));
    }
    comm->ThenWaitFor(comm_op->ThenRecordEvent(ctx));

    VLOG(1) << comm->DebugString() << " [" << comm_op->name() << "] [CastNIn]";
    functor::CastN<float, Eigen::half>()(inputs, comm_input_vec, ctx,
                                         comm->stream());
    VLOG(1) << comm->DebugString() << " [" << comm_op->name()
            << "] [AlltoallvN]";
    TF_RETURN_IF_ERROR(comm->AlltoallvN(*comm_input_vec, host_sizes,
                                        common_sizes, comm_output_vec));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name() << "] [CastNOut]";
    functor::CastN<Eigen::half, float>()(*comm_output_vec, outputs, ctx,
                                         comm->stream());
    return Status::OK();
  }
};

}  // namespace functor

template <typename DTYPE, typename WIRE_DTYPE>
class NcclAlltoallvNOp : public NcclCommAsyncOp {
 public:
  explicit NcclAlltoallvNOp(OpKernelConstruction* ctx) : NcclCommAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_columns", &num_columns_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("common_shapes", &common_shapes_));
    for (int64 i = 0; i < num_columns_; ++i) {
      int64 common_shape_size = 1;
      for (int64 dim = 0; dim < common_shapes_[i].dims(); ++dim) {
        common_shape_size *= common_shapes_[i].dim_size(dim);
      }
      common_sizes_.push_back(common_shape_size);
    }
  }

  virtual void ComputeAsyncWithComm(NcclComm* comm, OpKernelContext* ctx,
                                    DoneCallback done) override {
    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_on_host(true);
    host_alloc_attrs.set_gpu_compatible(true);

    std::vector<Tensor>* inputs_vec = new std::vector<Tensor>();
    std::vector<Tensor>* inputs_sizes_vec = new std::vector<Tensor>();
    std::vector<Tensor*>* outputs_sizes_vec = new std::vector<Tensor*>();
    Tensor* all_sizes = new Tensor();
    Tensor* host_all_sizes = new Tensor();
    Tensor* outputs_total_size = new Tensor();
    Tensor* host_outputs_total_size = new Tensor();
    Tensor* host_outputs_sizes_ptrs = new Tensor();
    std::vector<Tensor*>* comm_input_vec = new std::vector<Tensor*>();
    for (int i = 0; i < num_columns_; ++i) {
      comm_input_vec->push_back(new Tensor());
    }
    std::vector<Tensor*>* comm_output_vec = new std::vector<Tensor*>();
    for (int i = 0; i < num_columns_; ++i) {
      comm_output_vec->push_back(new Tensor());
    }
    auto done_ = [this, inputs_vec, inputs_sizes_vec, outputs_sizes_vec,
                  all_sizes, host_all_sizes, outputs_total_size,
                  host_outputs_total_size, host_outputs_sizes_ptrs,
                  comm_input_vec, comm_output_vec, done]() {
      delete inputs_vec;
      delete inputs_sizes_vec;
      delete outputs_sizes_vec;
      delete all_sizes;
      delete host_all_sizes;
      delete outputs_total_size;
      delete host_outputs_total_size;
      delete host_outputs_sizes_ptrs;
      for (int i = 0; i < num_columns_; ++i) {
        delete comm_input_vec->at(i);
      }
      delete comm_input_vec;
      for (int i = 0; i < num_columns_; ++i) {
        delete comm_output_vec->at(i);
      }
      delete comm_output_vec;
      done();
    };

    auto* ctx_stream = ctx->op_device_context()->stream();
    CudaStream ctx_cu_stream = CudaStream(ctx_stream);

    OpInputList inputs;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("inputs", &inputs), done_);
    for (int i = 0; i < num_columns_; ++i) {
      inputs_vec->push_back(inputs[i]);
    }

    OpInputList inputs_sizes;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("inputs_sizes", &inputs_sizes),
                         done_);
    for (int i = 0; i < num_columns_; ++i) {
      auto& input_sizes = inputs_sizes[i];
      OP_REQUIRES_ASYNC(
          ctx, input_sizes.NumElements() == comm->size(),
          errors::InvalidArgument(
              "Sizes of input ", i, " has ", input_sizes.NumElements(),
              " elements, which is not equal to communicator size: ",
              comm->size()),
          done_);
      inputs_sizes_vec->push_back(input_sizes);
    }

    for (int i = 0; i < num_columns_; ++i) {
      Tensor* outputs_sizes;
      OP_REQUIRES_OK_ASYNC(ctx,
                           ctx->allocate_output(num_columns_ + i,
                                                {comm->size()}, &outputs_sizes),
                           done_);
      outputs_sizes_vec->push_back(outputs_sizes);
    }

    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(
            DT_INT32, TensorShape({num_columns_ * comm->size() * comm->size()}),
            all_sizes),
        done_);

    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(
            DT_INT32, TensorShape({num_columns_ * comm->size() * comm->size()}),
            host_all_sizes, host_alloc_attrs),
        done_);

    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT32, TensorShape({num_columns_}),
                           outputs_total_size),
        done_);

    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT32, TensorShape({num_columns_}),
                           host_outputs_total_size, host_alloc_attrs),
        done_);

    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(
            DT_INT8,
            TensorShape({num_columns_ * static_cast<int32>(sizeof(int32*))}),
            host_outputs_sizes_ptrs, host_alloc_attrs),
        done_);

    comm->RunAsync(
        "NcclAlltoallvN", ctx, done_,
        [this, comm, ctx, ctx_cu_stream, done_, inputs_vec, inputs_sizes_vec,
         outputs_sizes_vec, all_sizes, host_all_sizes, outputs_total_size,
         host_outputs_total_size, host_outputs_sizes_ptrs, comm_input_vec,
         comm_output_vec]() {
          // Collect sizes of all inputs across devices.
          VLOG(1) << comm->DebugString() << " [" << name()
                  << "] [GroupAllgather]";
          OP_REQUIRES_OK_ASYNC(
              ctx, comm->GroupAllgather(*inputs_sizes_vec, all_sizes), done_);
          ThenWaitFor(ctx, comm->ThenRecordEvent());

          // Compute output sizes.
          const int32* d_all_sizes = all_sizes->flat<int32>().data();
          int32* d_outputs_total_size =
              const_cast<int32*>(outputs_total_size->flat<int32>().data());
          int32* h_outputs_total_size =
              const_cast<int32*>(host_outputs_total_size->flat<int32>().data());
          int32** hd_outputs_sizes = reinterpret_cast<int32**>(
              host_outputs_sizes_ptrs->flat<int8>().data());
          for (size_t i = 0; i < num_columns_; ++i) {
            hd_outputs_sizes[i] =
                outputs_sizes_vec->at(i)->flat<int32>().data();
          }
          ctx_cu_stream.ThenMemset(d_outputs_total_size, 0,
                                   num_columns_ * sizeof(int32));
          functor::SliceSumN<GPUDevice, int32>()(
              comm->size(), comm->size(), comm->rank(),
              outputs_sizes_vec->size(), d_all_sizes, d_outputs_total_size,
              hd_outputs_sizes, ctx->eigen_device<GPUDevice>());
          ThenCopyToHost(ctx, host_outputs_total_size, *outputs_total_size);
          ThenCopyToHost(ctx, host_all_sizes, *all_sizes);
          BlockHostUntilDone(ctx);

          // Allocate outputs.
          std::vector<Tensor*> outputs_vec;
          for (int n = 0; n < num_columns_; ++n) {
            TensorShape output_shape;
            PartialTensorShape({h_outputs_total_size[n]})
                .Concatenate(common_shapes_[n])
                .AsTensorShape(&output_shape);
            Tensor* output;
            OP_REQUIRES_OK_ASYNC(
                ctx, ctx->allocate_output(n, output_shape, &output), done_);
            outputs_vec.push_back(output);
          }

          // Cast and communicate.
          auto call = functor::NcclAlltoallvNCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              call(*inputs_vec, *host_all_sizes, common_sizes_, &outputs_vec,
                   comm_input_vec, comm_output_vec, ctx, comm, this),
              done_);
        });
  }

 private:
  int64 num_columns_;
  std::vector<PartialTensorShape> common_shapes_;
  std::vector<int64> common_sizes_;
};  // namespace functor

#define REGISTER_KERNEL(DTYPE, WIRE_DTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAlltoallvN")                       \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<DTYPE>("dtype")            \
                              .TypeConstraint<WIRE_DTYPE>("wire_dtype"), \
                          NcclAlltoallvNOp<DTYPE, WIRE_DTYPE>);
TF_CALL_NCCL_CAST_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

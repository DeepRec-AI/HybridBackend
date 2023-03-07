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

#include "hybridbackend/tensorflow/common/cast.h"
#include "hybridbackend/tensorflow/common/host_functions.h"
#include "hybridbackend/tensorflow/distribute/nccl/collective.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

#if GOOGLE_CUDA
namespace functor {
template <typename DTYPE, typename WIRE_DTYPE>
struct NcclAlltoallvCall {
  Status operator()(const Tensor& input, const int32* send_counts,
                    const int32* recv_counts, const int64 common_size,
                    Tensor* output, Tensor* comm_input, Tensor* comm_output,
                    CollectiveTopology topology, OpKernelContext* ctx,
                    NcclCollective* coll, NcclCollectiveAsyncOp* comm_op) {
    coll->stream()->ThenWaitUntilComputeDone(ctx);
    VLOG(1) << coll->DebugString() << " [" << comm_op->name()
            << "] [Alltoallv]";
    TF_RETURN_IF_ERROR(coll->Alltoallv(input, send_counts, recv_counts,
                                       common_size, output, topology));
    return Status::OK();
  }
};

template <>
struct NcclAlltoallvCall<float, Eigen::half> {
  Status operator()(const Tensor& input, const int32* send_counts,
                    const int32* recv_counts, const int64 common_size,
                    Tensor* output, Tensor* comm_input, Tensor* comm_output,
                    CollectiveTopology topology, OpKernelContext* ctx,
                    NcclCollective* coll, NcclCollectiveAsyncOp* comm_op) {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, input.shape(), comm_input));
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DT_HALF, output->shape(), comm_output));
    coll->stream()->ThenWaitUntilComputeDone(ctx);
    VLOG(1) << coll->DebugString() << " [" << comm_op->name() << "] [CastIn]";
    functor::Cast<float, Eigen::half>()(input, comm_input, ctx,
                                        coll->stream()->get());
    VLOG(1) << coll->DebugString() << " [" << comm_op->name()
            << "] [Alltoallv]";
    TF_RETURN_IF_ERROR(coll->Alltoallv(*comm_input, send_counts, recv_counts,
                                       common_size, comm_output, topology));
    VLOG(1) << coll->DebugString() << " [" << comm_op->name() << "] [CastOut]";
    functor::Cast<Eigen::half, float>()(*comm_output, output, ctx,
                                        coll->stream()->get());
    return Status::OK();
  }
};

template <typename DTYPE, typename WIRE_DTYPE>
struct NcclAlltoallvNCall {
  Status operator()(const std::vector<Tensor>& n_input,
                    const std::vector<Tensor*>& n_send_sizes,
                    const std::vector<Tensor*>& n_recv_sizes,
                    const std::vector<int64>& n_common_size,
                    std::vector<Tensor*>* n_output,
                    std::vector<Tensor*>* n_comm_input,
                    std::vector<Tensor*>* n_comm_output,
                    CollectiveTopology topology, OpKernelContext* ctx,
                    NcclCollective* coll, NcclCollectiveAsyncOp* comm_op) {
    std::vector<int32*> n_send_sizes_vec(n_input.size(), nullptr);
    std::vector<int32*> n_recv_sizes_vec(n_input.size(), nullptr);
    for (size_t idx = 0; idx < n_input.size(); ++idx) {
      n_send_sizes_vec[idx] = n_send_sizes[idx]->flat<int32>().data();
      n_recv_sizes_vec[idx] = n_recv_sizes[idx]->flat<int32>().data();
    }
    coll->stream()->ThenWaitUntilComputeDone(ctx);
    VLOG(1) << coll->DebugString() << " [" << comm_op->name()
            << "] [AlltoallvN]";
    TF_RETURN_IF_ERROR(coll->AlltoallvN(n_input, n_send_sizes_vec,
                                        n_recv_sizes_vec, n_common_size,
                                        n_output, topology));
    return Status::OK();
  }
};

template <>
struct NcclAlltoallvNCall<float, Eigen::half> {
  Status operator()(const std::vector<Tensor>& n_input,
                    const std::vector<Tensor*>& n_send_sizes,
                    const std::vector<Tensor*>& n_recv_sizes,
                    const std::vector<int64>& n_common_size,
                    std::vector<Tensor*>* n_output,
                    std::vector<Tensor*>* n_comm_input,
                    std::vector<Tensor*>* n_comm_output,
                    CollectiveTopology topology, OpKernelContext* ctx,
                    NcclCollective* coll, NcclCollectiveAsyncOp* comm_op) {
    std::vector<int32*> n_send_sizes_vec(n_input.size(), 0);
    std::vector<int32*> n_recv_sizes_vec(n_input.size(), 0);
    for (size_t idx = 0; idx < n_input.size(); ++idx) {
      n_send_sizes_vec[idx] = n_send_sizes[idx]->flat<int32>().data();
      n_recv_sizes_vec[idx] = n_recv_sizes[idx]->flat<int32>().data();
    }
    for (int idx = 0; idx < n_input.size(); ++idx) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, n_input[idx].shape(),
                                            n_comm_input->at(idx),
                                            ctx->input_alloc_attr(idx)));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, n_output->at(idx)->shape(),
                                            n_comm_output->at(idx),
                                            ctx->output_alloc_attr(idx)));
    }
    coll->stream()->ThenWaitUntilComputeDone(ctx);

    VLOG(1) << coll->DebugString() << " [" << comm_op->name() << "] [CastNIn]";
    functor::CastN<float, Eigen::half>()(n_input, n_comm_input, ctx,
                                         coll->stream()->get());
    VLOG(1) << coll->DebugString() << " [" << comm_op->name()
            << "] [AlltoallvN]";
    TF_RETURN_IF_ERROR(coll->AlltoallvN(*n_comm_input, n_send_sizes_vec,
                                        n_recv_sizes_vec, n_common_size,
                                        n_comm_output, topology));
    VLOG(1) << coll->DebugString() << " [" << comm_op->name() << "] [CastNOut]";
    functor::CastN<Eigen::half, float>()(*n_comm_output, n_output, ctx,
                                         coll->stream()->get());
    return Status::OK();
  }
};

}  // namespace functor
#endif  // GOOGLE_CUDA

REGISTER_OP("HbNcclAlltoallv")
    .Output("output: dtype")
    .Output("output_sizes: int32")
    .Input("handle: resource")
    .Input("input: dtype")
    .Input("input_sizes: int32")
    .Attr("common_shape: shape = {}")
    .Attr("topology: int = 0")
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
AllToAllv using a NCCL communicator.

output: An exchanged tensor.
output_sizes: An tensor for 1st dim of the output.
handle: Handle of a NCCL communicator.
input: A tensor to exchange.
input_sizes: A tensor for 1st dim of input.
)doc");

#if GOOGLE_CUDA
template <typename DTYPE, typename WIRE_DTYPE>
class NcclAlltoallvOp : public NcclCollectiveAsyncOp {
 public:
  explicit NcclAlltoallvOp(OpKernelConstruction* ctx)
      : NcclCollectiveAsyncOp(ctx) {
    int topology;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("topology", &topology));
    topology_ = static_cast<CollectiveTopology>(topology);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("common_shape", &common_shape_));
    common_size_ = 1;
    for (int64 dim = 0; dim < common_shape_.dims(); ++dim) {
      common_size_ *= common_shape_.dim_size(dim);
    }
  }

  virtual void CollectiveComputeAsync(NcclCollective* coll,
                                      OpKernelContext* ctx,
                                      DoneCallback done) override {
    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_on_host(true);
    host_alloc_attrs.set_gpu_compatible(true);

    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);
    const Tensor* input_sizes;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input_sizes", &input_sizes), done);
    int active_size = coll->compute_active_size(topology_);
    TensorShape output_sizes_shape({active_size});
    Tensor* output_sizes;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(1, output_sizes_shape, &output_sizes), done);

    Tensor* host_input_sizes = new Tensor();
    Tensor* host_output_sizes = new Tensor();
    Tensor* comm_input = new Tensor();
    Tensor* comm_output = new Tensor();
    auto done_ = [host_input_sizes, host_output_sizes, comm_input, comm_output,
                  done]() {
      delete host_input_sizes;
      delete host_output_sizes;
      delete comm_input;
      delete comm_output;
      done();
    };

    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT32, TensorShape({active_size}),
                           host_input_sizes, host_alloc_attrs),
        done_);
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT32, TensorShape({active_size}),
                           host_output_sizes, host_alloc_attrs),
        done_);
    coll->stream()->LaunchUntilComputeDone(
        ctx,
        [input_sizes, output_sizes, host_input_sizes, host_output_sizes, input,
         comm_input, comm_output, active_size, ctx, coll, this, done_]() {
          ctx->op_device_context()->stream()->ThenMemcpy(
              const_cast<char*>(host_input_sizes->tensor_data().data()),
              se::DeviceMemoryBase(
                  const_cast<char*>(input_sizes->tensor_data().data()),
                  input_sizes->TotalBytes()),
              host_input_sizes->TotalBytes());
          // Collect sizes of all inputs across devices.
          VLOG(1) << coll->DebugString() << " [" << name() << "] [Alltoall]";
          OP_REQUIRES_OK_ASYNC(
              ctx, coll->Alltoall(*input_sizes, output_sizes, topology_),
              done_);
          coll->stream()->BlockComputeUntilDone(ctx);
          ctx->op_device_context()->stream()->ThenMemcpy(
              const_cast<char*>(host_output_sizes->tensor_data().data()),
              se::DeviceMemoryBase(
                  const_cast<char*>(output_sizes->tensor_data().data()),
                  output_sizes->TotalBytes()),
              host_output_sizes->TotalBytes());
          ctx->op_device_context()->stream()->BlockHostUntilDone();

          // Allocate output.
          int32 total_output_size = 0;
          for (int32 rank = 0; rank < active_size; ++rank) {
            total_output_size += host_output_sizes->flat<int32>()(rank);
          }
          TensorShape output_shape;
          PartialTensorShape({total_output_size})
              .Concatenate(common_shape_)
              .AsTensorShape(&output_shape);
          Tensor* output;
          OP_REQUIRES_OK_ASYNC(
              ctx, ctx->allocate_output(0, output_shape, &output), done_);

          // Cast and communicate.
          auto call = functor::NcclAlltoallvCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              call(*input, host_input_sizes->flat<int32>().data(),
                   host_output_sizes->flat<int32>().data(), common_size_,
                   output, comm_input, comm_output, topology_, ctx, coll, this),
              done_);
          coll->stream()->BlockComputeUntilDone(ctx, done_);
        });
  }

 private:
  PartialTensorShape common_shape_;
  int64 common_size_;
  CollectiveTopology topology_;
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

REGISTER_OP("HbNcclAlltoallvN")
    .Output("n_output: N * dtype")
    .Output("n_output_sizes: N * int32")
    .Input("handle: resource")
    .Input("n_input: N * dtype")
    .Input("n_input_sizes: N * int32")
    .Attr("N: int >= 1 = 1")
    .Attr("common_shape: list(shape)")
    .Attr("topology: int = 0")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("wire_dtype: {" TF_OP_NCCL_WIRE_DTYPE_LIST "}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      std::vector<PartialTensorShape> common_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("common_shape", &common_shape));
      for (int64 n = 0; n < N; ++n) {
        shape_inference::ShapeHandle shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(common_shape[n], &shape));
        TF_RETURN_IF_ERROR(c->Concatenate(
            c->Vector(shape_inference::InferenceContext::kUnknownDim), shape,
            &shape));
        c->set_output(n, shape);
        c->set_output(N + n, c->input(1 + N + n));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Packed AllToAllv using a NCCL communicator.

n_output: N exchanged tensors.
n_output_sizes: N tensors for 1st dim of outputs.
handle: Handle of a NCCL communicator.
n_input: N tensors to exchange.
n_input_sizes: N tensors for 1st dim of inputs.
)doc");

#if GOOGLE_CUDA
template <typename DTYPE, typename WIRE_DTYPE>
class NcclAlltoallvNOp : public NcclCollectiveAsyncOp {
 public:
  explicit NcclAlltoallvNOp(OpKernelConstruction* ctx)
      : NcclCollectiveAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    int topology;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("topology", &topology));
    topology_ = static_cast<CollectiveTopology>(topology);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("common_shape", &common_shape_));
    for (int64 idx = 0; idx < N_; ++idx) {
      int64 common_shape_size = 1;
      for (int64 dim = 0; dim < common_shape_[idx].dims(); ++dim) {
        common_shape_size *= common_shape_[idx].dim_size(dim);
      }
      common_sizes_.push_back(common_shape_size);
    }
  }

  virtual void CollectiveComputeAsync(NcclCollective* coll,
                                      OpKernelContext* ctx,
                                      DoneCallback done) override {
    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_on_host(true);
    host_alloc_attrs.set_gpu_compatible(true);

    std::vector<Tensor>* n_input = new std::vector<Tensor>();
    std::vector<Tensor>* n_input_sizes = new std::vector<Tensor>();
    std::vector<Tensor*>* n_output_sizes = new std::vector<Tensor*>();
    std::vector<Tensor*>* n_host_input_sizes = new std::vector<Tensor*>();
    std::vector<Tensor*>* n_host_output_sizes = new std::vector<Tensor*>();
    std::vector<Tensor*>* n_comm_input = new std::vector<Tensor*>();
    std::vector<Tensor*>* n_comm_output = new std::vector<Tensor*>();
    for (int idx = 0; idx < N_; ++idx) {
      n_host_input_sizes->push_back(new Tensor());
      n_host_output_sizes->push_back(new Tensor());
      n_comm_input->push_back(new Tensor());
      n_comm_output->push_back(new Tensor());
    }
    auto done_ = [this, n_input, n_input_sizes, n_output_sizes,
                  n_host_input_sizes, n_host_output_sizes, n_comm_input,
                  n_comm_output, done]() {
      delete n_input;
      delete n_input_sizes;
      delete n_output_sizes;
      for (int idx = 0; idx < N_; ++idx) {
        delete n_host_input_sizes->at(idx);
        delete n_host_output_sizes->at(idx);
        delete n_comm_input->at(idx);
        delete n_comm_output->at(idx);
      }
      delete n_host_input_sizes;
      delete n_host_output_sizes;
      delete n_comm_input;
      delete n_comm_output;
      done();
    };

    OpInputList n_input_list;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("n_input", &n_input_list), done_);
    OpInputList n_input_sizes_list;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->input_list("n_input_sizes", &n_input_sizes_list), done_);

    int active_size = coll->compute_active_size(topology_);

    for (int idx = 0; idx < N_; ++idx) {
      n_input->push_back(n_input_list[idx]);
      auto& input_sizes = n_input_sizes_list[idx];
      OP_REQUIRES_ASYNC(
          ctx, input_sizes.NumElements() == active_size,
          errors::InvalidArgument(
              "Sizes of input ", idx, " has ", input_sizes.NumElements(),
              " elements, which is not equal to active rank size: ",
              active_size),
          done_);
      n_input_sizes->push_back(input_sizes);
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(DT_INT32, TensorShape({active_size}),
                             n_host_input_sizes->at(idx), host_alloc_attrs),
          done);
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(DT_INT32, TensorShape({active_size}),
                             n_host_output_sizes->at(idx), host_alloc_attrs),
          done);
      Tensor* output_sizes;
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(N_ + idx, {active_size}, &output_sizes),
          done_);
      n_output_sizes->push_back(output_sizes);
    }

    coll->stream()->LaunchUntilComputeDone(
        ctx, [this, coll, ctx, done_, n_input, n_input_sizes, n_output_sizes,
              n_host_input_sizes, n_host_output_sizes, n_comm_input,
              n_comm_output, active_size]() {
          for (size_t idx = 0; idx < N_; ++idx) {
            ctx->op_device_context()->stream()->ThenMemcpy(
                const_cast<char*>(
                    n_host_input_sizes->at(idx)->tensor_data().data()),
                se::DeviceMemoryBase(
                    const_cast<char*>(
                        n_input_sizes->at(idx).tensor_data().data()),
                    n_input_sizes->at(idx).TotalBytes()),
                n_host_input_sizes->at(idx)->TotalBytes());
          }
          VLOG(1) << coll->DebugString() << " [" << name() << "] [AlltoallN]";
          OP_REQUIRES_OK_ASYNC(
              ctx, coll->AlltoallN(*n_input_sizes, n_output_sizes, topology_),
              done_);
          coll->stream()->BlockComputeUntilDone(ctx);
          for (size_t idx = 0; idx < N_; ++idx) {
            ctx->op_device_context()->stream()->ThenMemcpy(
                const_cast<char*>(
                    n_host_output_sizes->at(idx)->tensor_data().data()),
                se::DeviceMemoryBase(
                    const_cast<char*>(
                        n_output_sizes->at(idx)->tensor_data().data()),
                    n_output_sizes->at(idx)->TotalBytes()),
                n_host_output_sizes->at(idx)->TotalBytes());
          }
          ctx->op_device_context()->stream()->BlockHostUntilDone();
          std::vector<int32> n_host_output_total_size(N_, 0);
          for (size_t idx = 0; idx < N_; ++idx) {
            n_host_output_total_size[idx] = 0;
            for (int32 rank = 0; rank < active_size; ++rank) {
              n_host_output_total_size[idx] +=
                  n_host_output_sizes->at(idx)->flat<int32>()(rank);
            }
          }

          std::vector<Tensor*> n_output;
          for (int idx = 0; idx < N_; ++idx) {
            TensorShape output_shape;
            PartialTensorShape({n_host_output_total_size[idx]})
                .Concatenate(common_shape_[idx])
                .AsTensorShape(&output_shape);
            Tensor* output;
            OP_REQUIRES_OK_ASYNC(
                ctx, ctx->allocate_output(idx, output_shape, &output), done_);
            n_output.push_back(output);
          }

          auto call = functor::NcclAlltoallvNCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              call(*n_input, *n_host_input_sizes, *n_host_output_sizes,
                   common_sizes_, &n_output, n_comm_input, n_comm_output,
                   topology_, ctx, coll, this),
              done_);
          coll->stream()->BlockComputeUntilDone(ctx, done_);
        });
  }

 private:
  CollectiveTopology topology_;
  int64 N_;
  std::vector<PartialTensorShape> common_shape_;
  std::vector<int64> common_sizes_;
};

#define REGISTER_KERNEL(DTYPE, WIRE_DTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAlltoallvN")                       \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<DTYPE>("dtype")            \
                              .TypeConstraint<WIRE_DTYPE>("wire_dtype"), \
                          NcclAlltoallvNOp<DTYPE, WIRE_DTYPE>);
TF_CALL_NCCL_CAST_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

REGISTER_OP("HbNcclAlltoallvMergedN")
    .Output("n_output: N * dtype")
    .Output("n_output_sizes: N * int32")
    .Input("handle: resource")
    .Input("n_input: N * dtype")
    .Input("n_input_sizes: N * int32")
    .Attr("N: int >= 1 = 1")
    .Attr("common_shape: shape = {}")
    .Attr("topology: int = 0")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("wire_dtype: {" TF_OP_NCCL_WIRE_DTYPE_LIST "}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));

      PartialTensorShape common_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("common_shape", &common_shape));
      for (int64 n = 0; n < N; ++n) {
        shape_inference::ShapeHandle shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(common_shape, &shape));
        TF_RETURN_IF_ERROR(c->Concatenate(
            c->Vector(shape_inference::InferenceContext::kUnknownDim), shape,
            &shape));
        c->set_output(n, shape);
        c->set_output(N + n, c->input(1 + N + n));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Packed merged AllToAllv using a NCCL communicator.

n_output: N exchanged tensors.
n_output_sizes: N tensors for 1st dim of outputs.
handle: Handle of a NCCL communicator.
n_input: N tensors to exchange.
n_input_sizes: N tensors for 1st dim of inputs.
)doc");

#if GOOGLE_CUDA
template <typename DTYPE, typename WIRE_DTYPE>
class NcclAlltoallvMergedNOp : public NcclCollectiveAsyncOp {
 public:
  explicit NcclAlltoallvMergedNOp(OpKernelConstruction* ctx)
      : NcclCollectiveAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    int topology;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("topology", &topology));
    topology_ = static_cast<CollectiveTopology>(topology);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("common_shape", &common_shape_));
    common_size_ = 1;
    for (int64 dim = 0; dim < common_shape_.dims(); ++dim) {
      common_size_ *= common_shape_.dim_size(dim);
    }
  }

  virtual void CollectiveComputeAsync(NcclCollective* coll,
                                      OpKernelContext* ctx,
                                      DoneCallback done) override {
    // FIXME: Random crash happens
    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_on_host(true);
    host_alloc_attrs.set_gpu_compatible(true);

    std::vector<Tensor>* n_input_sizes = new std::vector<Tensor>();
    std::vector<Tensor*>* n_output_sizes = new std::vector<Tensor*>();
    Tensor* buffer_input = new Tensor();
    Tensor* buffer_output = new Tensor();
    Tensor* buffer_comm_input = new Tensor();
    Tensor* buffer_comm_output = new Tensor();
    Tensor* host_sizes_buffer = new Tensor();
    auto done_ = [n_input_sizes, n_output_sizes, buffer_input, buffer_output,
                  buffer_comm_input, buffer_comm_output, host_sizes_buffer,
                  done]() {
      delete n_input_sizes;
      delete n_output_sizes;
      delete buffer_input;
      delete buffer_output;
      delete buffer_comm_input;
      delete buffer_comm_output;
      delete host_sizes_buffer;
      done();
    };

    OpInputList n_input_list;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("n_input", &n_input_list), done_);
    OpInputList n_input_sizes_list;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->input_list("n_input_sizes", &n_input_sizes_list), done_);
    std::vector<int64> input_bytes_vec(N_, 0);
    int64 total_input_bytes = 0;
    int active_size = coll->compute_active_size(topology_);
    for (int idx = 0; idx < N_; ++idx) {
      auto& input_sizes = n_input_sizes_list[idx];
      OP_REQUIRES_ASYNC(
          ctx, input_sizes.NumElements() == active_size,
          errors::InvalidArgument(
              "Sizes of input ", idx, " has ", input_sizes.NumElements(),
              " elements, which is not equal to active rank size: ",
              active_size),
          done_);
      n_input_sizes->push_back(input_sizes);
      Tensor* output_sizes;
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(N_ + idx, {active_size}, &output_sizes),
          done_);
      n_output_sizes->push_back(output_sizes);
      const auto input_bytes = n_input_list[idx].TotalBytes();
      total_input_bytes += input_bytes;
      input_bytes_vec[idx] = input_bytes;
    }
    OP_REQUIRES_OK_ASYNC(ctx,
                         ctx->allocate_temp(DataTypeToEnum<DTYPE>::value,
                                            {total_input_bytes}, buffer_input),
                         done_);
    int64 input_offset_bytes = 0;
    for (int64 idx = 0; idx < N_; ++idx) {
      se::DeviceMemoryBase input_dst_ptr(
          const_cast<char*>(buffer_input->tensor_data().data()) +
              input_offset_bytes,
          input_bytes_vec[idx]);
      ctx->op_device_context()->stream()->ThenMemcpy(
          &input_dst_ptr,
          se::DeviceMemoryBase(
              const_cast<char*>(n_input_list[idx].tensor_data().data()),
              input_bytes_vec[idx]),
          input_bytes_vec[idx]);
      input_offset_bytes += input_bytes_vec[idx];
    }

    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT32, {(2 * N_ + 2) * active_size + N_},
                           host_sizes_buffer, host_alloc_attrs),
        done_);
    int32* host_input_sizes = host_sizes_buffer->flat<int32>().data();
    int32* host_output_sizes =
        host_sizes_buffer->flat<int32>().data() + N_ * active_size;
    int32* host_input_total_sizes =
        host_sizes_buffer->flat<int32>().data() + 2 * N_ * active_size;
    int32* host_output_total_sizes =
        host_sizes_buffer->flat<int32>().data() + (2 * N_ + 1) * active_size;
    int32* host_n_output_sizes =
        host_sizes_buffer->flat<int32>().data() + (2 * N_ + 2) * active_size;

    coll->stream()->LaunchUntilComputeDone(
        ctx,
        [n_input_sizes, n_output_sizes, host_input_sizes, host_output_sizes,
         host_input_total_sizes, host_output_total_sizes, host_n_output_sizes,
         buffer_input, buffer_output, buffer_comm_input, buffer_comm_output,
         active_size, coll, ctx, done_, this]() {
          VLOG(1) << coll->DebugString() << " [" << name() << "] [AlltoallN]";
          OP_REQUIRES_OK_ASYNC(
              ctx, coll->AlltoallN(*n_input_sizes, n_output_sizes, topology_),
              done_);
          coll->stream()->BlockComputeUntilDone(ctx);
          for (int64 idx = 0; idx < N_; ++idx) {
            ctx->op_device_context()->stream()->ThenMemcpy(
                host_input_sizes + idx * active_size,
                se::DeviceMemoryBase(
                    n_input_sizes->at(idx).flat<int32>().data(),
                    n_input_sizes->at(idx).TotalBytes()),
                n_input_sizes->at(idx).TotalBytes());
            ctx->op_device_context()->stream()->ThenMemcpy(
                host_output_sizes + idx * active_size,
                se::DeviceMemoryBase(
                    n_output_sizes->at(idx)->flat<int32>().data(),
                    n_output_sizes->at(idx)->TotalBytes()),
                n_output_sizes->at(idx)->TotalBytes());
          }
          ctx->op_device_context()->stream()->BlockHostUntilDone();
          for (int64 rank = 0; rank < active_size; ++rank) {
            host_input_total_sizes[rank] = 0;
            host_output_total_sizes[rank] = 0;
            for (int64 idx = 0; idx < N_; ++idx) {
              host_input_total_sizes[rank] +=
                  host_input_sizes[idx * active_size + rank];
              host_output_total_sizes[rank] +=
                  host_output_sizes[idx * active_size + rank];
            }
          }
          for (int64 idx = 0; idx < N_; ++idx) {
            host_n_output_sizes[idx] = 0;
            for (int64 rank = 0; rank < active_size; ++rank) {
              host_n_output_sizes[idx] +=
                  host_output_sizes[idx * active_size + rank];
            }
          }

          int64 total_output_sizes = 0;
          for (int rank = 0; rank < active_size; ++rank) {
            total_output_sizes += host_output_total_sizes[rank];
          }
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DataTypeToEnum<DTYPE>::value,
                                 {common_size_ * total_output_sizes},
                                 buffer_output),
              done_);

          auto call = functor::NcclAlltoallvCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              call(*buffer_input, host_input_total_sizes,
                   host_output_total_sizes, common_size_, buffer_output,
                   buffer_comm_input, buffer_comm_output, topology_, ctx, coll,
                   this),
              done_);
          coll->stream()->BlockComputeUntilDone(ctx);

          std::vector<Tensor*> n_output;
          for (int idx = 0; idx < N_; ++idx) {
            TensorShape output_shape;
            PartialTensorShape({host_n_output_sizes[idx]})
                .Concatenate(common_shape_)
                .AsTensorShape(&output_shape);
            Tensor* output;
            OP_REQUIRES_OK_ASYNC(
                ctx, ctx->allocate_output(idx, output_shape, &output), done_);
            n_output.push_back(output);
          }
          int64 output_offset_bytes = 0;
          for (int idx = 0; idx < N_; ++idx) {
            se::DeviceMemoryBase output_dst_ptr(
                const_cast<char*>(n_output[idx]->tensor_data().data()),
                host_n_output_sizes[idx]);
            ctx->op_device_context()->stream()->ThenMemcpy(
                &output_dst_ptr,
                se::DeviceMemoryBase(
                    const_cast<char*>(buffer_output->tensor_data().data()) +
                        output_offset_bytes,
                    host_n_output_sizes[idx]),
                host_n_output_sizes[idx]);
            output_offset_bytes += host_n_output_sizes[idx] * common_size_ *
                                   DataTypeSize(DataTypeToEnum<DTYPE>::value);
          }
          coll->stream()->BlockComputeUntilDone(ctx, done_);
        });
  }

 private:
  CollectiveTopology topology_;
  int64 N_;
  PartialTensorShape common_shape_;
  int64 common_size_;
};

#define REGISTER_KERNEL(DTYPE, WIRE_DTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAlltoallvMergedN")                 \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<DTYPE>("dtype")            \
                              .TypeConstraint<WIRE_DTYPE>("wire_dtype"), \
                          NcclAlltoallvMergedNOp<DTYPE, WIRE_DTYPE>);
TF_CALL_NCCL_CAST_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif  // HYBRIDBACKEND_NCCL

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

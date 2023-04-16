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
#include "hybridbackend/tensorflow/distribute/nccl/collective.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

#if GOOGLE_CUDA
namespace functor {
template <typename DTYPE, typename WIRE_DTYPE>
struct NcclAlltoallCall {
  Status operator()(const Tensor& input, Tensor* output, Tensor* comm_input,
                    Tensor* comm_output, CollectiveTopology topology,
                    OpKernelContext* ctx, NcclCollective* coll,
                    NcclCollectiveAsyncOp* comm_op) {
    coll->stream()->ThenWaitUntilComputeDone(ctx);
    VLOG(1) << "[" << ctx->step_id() << "]" << coll->DebugString() << " ["
            << comm_op->name() << "] [Alltoall] ["
            << DataTypeString(input.dtype()) << "] ["
            << CollectiveTopologyString(topology) << "] (" << input.TotalBytes()
            << "B)";
    TF_RETURN_IF_ERROR(coll->Alltoall(input, output, topology));
    return Status::OK();
  }
};

template <>
struct NcclAlltoallCall<float, Eigen::half> {
  Status operator()(const Tensor& input, Tensor* output, Tensor* comm_input,
                    Tensor* comm_output, CollectiveTopology topology,
                    OpKernelContext* ctx, NcclCollective* coll,
                    NcclCollectiveAsyncOp* comm_op) {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, input.shape(), comm_input));
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DT_HALF, output->shape(), comm_output));
    coll->stream()->ThenWaitUntilComputeDone(ctx);
    VLOG(1) << "[" << ctx->step_id() << "]" << coll->DebugString() << " ["
            << comm_op->name() << "] [CastIn] ["
            << DataTypeString(input.dtype()) << "] (" << input.TotalBytes()
            << "B)";
    functor::Cast<float, Eigen::half>()(input, comm_input, ctx,
                                        coll->stream()->get());
    VLOG(1) << "[" << ctx->step_id() << "]" << coll->DebugString() << " ["
            << comm_op->name() << "] [Alltoall] ["
            << DataTypeString(comm_input->dtype()) << "] ["
            << CollectiveTopologyString(topology) << "] ("
            << comm_input->TotalBytes() << "B)";
    TF_RETURN_IF_ERROR(coll->Alltoall(*comm_input, comm_output, topology));
    VLOG(1) << "[" << ctx->step_id() << "]" << coll->DebugString() << " ["
            << comm_op->name() << "] [CastOut] ["
            << DataTypeString(output->dtype()) << "] (" << output->TotalBytes()
            << "B)";
    functor::Cast<Eigen::half, float>()(*output, comm_output, ctx,
                                        coll->stream()->get());
    return Status::OK();
  }
};

template <typename DTYPE, typename WIRE_DTYPE>
struct NcclAlltoallNCall {
  Status operator()(const std::vector<Tensor>& n_input,
                    std::vector<Tensor*>* n_output,
                    std::vector<Tensor*>* n_comm_input,
                    std::vector<Tensor*>* n_comm_output,
                    CollectiveTopology topology, OpKernelContext* ctx,
                    NcclCollective* coll, NcclCollectiveAsyncOp* comm_op) {
    coll->stream()->ThenWaitUntilComputeDone(ctx);
    if (VLOG_IS_ON(1)) {
      size_t input_total_bytes = 0;
      for (size_t idx = 0; idx < n_input.size(); ++idx) {
        input_total_bytes += n_input[idx].TotalBytes();
      }
      VLOG(1) << "[" << ctx->step_id() << "]" << coll->DebugString() << " ["
              << comm_op->name() << "] [AlltoallN] ["
              << DataTypeString(n_input[0].dtype()) << "] ["
              << CollectiveTopologyString(topology) << "] (" << n_input.size()
              << " inputs, " << input_total_bytes << "B)";
    }
    TF_RETURN_IF_ERROR(coll->AlltoallN(n_input, n_output, topology));
    return Status::OK();
  }
};

template <>
struct NcclAlltoallNCall<float, Eigen::half> {
  Status operator()(const std::vector<Tensor>& n_input,
                    std::vector<Tensor*>* n_output,
                    std::vector<Tensor*>* n_comm_input,
                    std::vector<Tensor*>* n_comm_output,
                    CollectiveTopology topology, OpKernelContext* ctx,
                    NcclCollective* coll, NcclCollectiveAsyncOp* comm_op) {
    for (int idx = 0; idx < n_input.size(); ++idx) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, n_input[idx].shape(),
                                            n_comm_input->at(idx),
                                            ctx->input_alloc_attr(idx)));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, n_output->at(idx)->shape(),
                                            n_comm_output->at(idx),
                                            ctx->output_alloc_attr(idx)));
    }
    coll->stream()->ThenWaitUntilComputeDone(ctx);
    if (VLOG_IS_ON(1)) {
      size_t input_total_bytes = 0;
      for (size_t idx = 0; idx < n_input.size(); ++idx) {
        input_total_bytes += n_input[idx].TotalBytes();
      }
      VLOG(1) << "[" << ctx->step_id() << "]" << coll->DebugString() << " ["
              << comm_op->name() << "] [CastIn] ["
              << DataTypeString(n_input[0].dtype()) << "] (" << n_input.size()
              << " inputs, " << input_total_bytes << "B)";
    }
    functor::CastN<float, Eigen::half>()(n_input, n_comm_input, ctx,
                                         coll->stream()->get());
    if (VLOG_IS_ON(1)) {
      size_t input_total_bytes = 0;
      for (size_t idx = 0; idx < n_comm_input->size(); ++idx) {
        input_total_bytes += n_comm_input->at(idx)->TotalBytes();
      }
      VLOG(1) << "[" << ctx->step_id() << "]" << coll->DebugString() << " ["
              << comm_op->name() << "] [AlltoallN] ["
              << DataTypeString(n_comm_input->at(0)->dtype()) << "] ["
              << CollectiveTopologyString(topology) << "] ("
              << n_comm_input->size() << " inputs, " << input_total_bytes
              << "B)";
    }
    TF_RETURN_IF_ERROR(coll->AlltoallN(*n_comm_input, n_comm_output, topology));
    if (VLOG_IS_ON(1)) {
      size_t input_total_bytes = 0;
      for (size_t idx = 0; idx < n_comm_output->size(); ++idx) {
        input_total_bytes += n_comm_output->at(idx)->TotalBytes();
      }
      VLOG(1) << "[" << ctx->step_id() << "]" << coll->DebugString() << " ["
              << comm_op->name() << "] [CastOut] ["
              << DataTypeString(n_comm_output->at(0)->dtype()) << "] ("
              << n_comm_output->size() << " inputs, " << input_total_bytes
              << "B)";
    }
    functor::CastN<Eigen::half, float>()(*n_comm_output, n_output, ctx,
                                         coll->stream()->get());
    return Status::OK();
  }
};
}  // namespace functor
#endif  // GOOGLE_CUDA

REGISTER_OP("HbNcclAlltoall")
    .Output("output: dtype")
    .Input("handle: resource")
    .Input("input: dtype")
    .Attr("topology: int = 0")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("wire_dtype: {" TF_OP_NCCL_WIRE_DTYPE_LIST "}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
AllToAll using a NCCL communicator.

output: Exchanged tensor.
handle: Handle of a NCCL communicator.
input: Tensor to exchange.
)doc");

#if GOOGLE_CUDA
template <typename DTYPE, typename WIRE_DTYPE>
class NcclAlltoallOp : public NcclCollectiveAsyncOp {
 public:
  explicit NcclAlltoallOp(OpKernelConstruction* ctx)
      : NcclCollectiveAsyncOp(ctx) {
    int topology;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("topology", &topology));
    topology_ = static_cast<CollectiveTopology>(topology);
  }

  virtual void CollectiveComputeAsync(NcclCollective* coll,
                                      OpKernelContext* ctx,
                                      DoneCallback done) override {
    Tensor* comm_input = new Tensor();
    Tensor* comm_output = new Tensor();
    auto done_ = [comm_input, comm_output, done]() {
      delete comm_input;
      delete comm_output;
      done();
    };

    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done_);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input->shape(), &output),
                         done_);
    coll->stream()->LaunchUntilComputeDone(
        ctx,
        [input, output, comm_input, comm_output, ctx, coll, this, done_]() {
          auto call = functor::NcclAlltoallCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(ctx,
                               call(*input, output, comm_input, comm_output,
                                    topology_, ctx, coll, this),
                               done_);
          coll->stream()->BlockComputeUntilDone(ctx, done_);
        });
  }

 private:
  CollectiveTopology topology_;
};

#define REGISTER_KERNEL(DTYPE, WIRE_DTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAlltoall")                         \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<DTYPE>("dtype")            \
                              .TypeConstraint<WIRE_DTYPE>("wire_dtype"), \
                          NcclAlltoallOp<DTYPE, WIRE_DTYPE>);
TF_CALL_NCCL_CAST_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

REGISTER_OP("HbNcclAlltoallN")
    .Output("n_output: N * dtype")
    .Input("handle: resource")
    .Input("n_input: N * dtype")
    .Attr("topology: int = 0")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("wire_dtype: {" TF_OP_NCCL_WIRE_DTYPE_LIST "}")
    .Attr("N: int >= 1 = 1")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      for (int64 n = 0; n < N; ++n) {
        c->set_output(n, c->input(1 + n));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Packed AllToAll using a NCCL communicator.

n_output: N exchanged tensors.
handle: Handle of a NCCL communicator.
n_input: N tensors to exchange.
)doc");

#if GOOGLE_CUDA
template <typename DTYPE, typename WIRE_DTYPE>
class NcclAlltoallNOp : public NcclCollectiveAsyncOp {
 public:
  explicit NcclAlltoallNOp(OpKernelConstruction* ctx)
      : NcclCollectiveAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    int topology;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("topology", &topology));
    topology_ = static_cast<CollectiveTopology>(topology);
  }

  virtual void CollectiveComputeAsync(NcclCollective* coll,
                                      OpKernelContext* ctx,
                                      DoneCallback done) override {
    std::vector<Tensor>* n_input = new std::vector<Tensor>();
    std::vector<Tensor*>* n_comm_input = new std::vector<Tensor*>();
    std::vector<Tensor*>* n_comm_output = new std::vector<Tensor*>();
    for (int idx = 0; idx < N_; ++idx) {
      n_comm_input->push_back(new Tensor());
      n_comm_output->push_back(new Tensor());
    }
    std::vector<Tensor*>* n_output = new std::vector<Tensor*>();

    auto done_ = [this, n_input, n_output, n_comm_input, n_comm_output,
                  done]() {
      delete n_input;
      delete n_output;
      for (int idx = 0; idx < N_; ++idx) {
        delete n_comm_input->at(idx);
        delete n_comm_output->at(idx);
      }
      delete n_comm_input;
      delete n_comm_output;
      done();
    };

    OpInputList n_input_list;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("n_input", &n_input_list), done_);
    for (int idx = 0; idx < N_; ++idx) {
      n_input->push_back(n_input_list[idx]);
      Tensor* output;
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(idx, n_input_list[idx].shape(), &output),
          done_);
      n_output->push_back(output);
    }

    coll->stream()->LaunchUntilComputeDone(
        ctx, [n_input, n_output, n_comm_input, n_comm_output, ctx, coll, this,
              done_]() {
          auto call = functor::NcclAlltoallNCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(ctx,
                               call(*n_input, n_output, n_comm_input,
                                    n_comm_output, topology_, ctx, coll, this),
                               done_);
          coll->stream()->BlockComputeUntilDone(ctx, done_);
        });
  }

 private:
  CollectiveTopology topology_;
  int64 N_;
};

#define REGISTER_KERNEL(DTYPE, WIRE_DTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAlltoallN")                        \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<DTYPE>("dtype")            \
                              .TypeConstraint<WIRE_DTYPE>("wire_dtype"), \
                          NcclAlltoallNOp<DTYPE, WIRE_DTYPE>);
TF_CALL_NCCL_CAST_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

REGISTER_OP("HbNcclAlltoallMergedN")
    .Output("n_output: N * dtype")
    .Input("handle: resource")
    .Input("n_input: N * dtype")
    .Attr("topology: int = 0")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("wire_dtype: {" TF_OP_NCCL_WIRE_DTYPE_LIST "}")
    .Attr("N: int >= 1 = 1")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      for (int64 n = 0; n < N; ++n) {
        c->set_output(n, c->input(1 + n));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Packed merged AllToAll using a NCCL communicator.

n_output: N exchanged tensors.
handle: Handle of a NCCL communicator.
n_input: N tensors to exchange.
)doc");

#if GOOGLE_CUDA
template <typename DTYPE, typename WIRE_DTYPE>
class NcclAlltoallMergedNOp : public NcclCollectiveAsyncOp {
 public:
  explicit NcclAlltoallMergedNOp(OpKernelConstruction* ctx)
      : NcclCollectiveAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    int topology;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("topology", &topology));
    topology_ = static_cast<CollectiveTopology>(topology);
  }

  virtual void CollectiveComputeAsync(NcclCollective* coll,
                                      OpKernelContext* ctx,
                                      DoneCallback done) override {
    std::vector<int64>* input_bytes_vec = new std::vector<int64>(N_, 0);
    std::vector<Tensor*>* n_output = new std::vector<Tensor*>(N_, nullptr);
    Tensor* buffer_input = new Tensor();
    Tensor* buffer_output = new Tensor();
    Tensor* buffer_comm_input = new Tensor();
    Tensor* buffer_comm_output = new Tensor();
    auto done_ = [input_bytes_vec, n_output, buffer_input, buffer_output,
                  buffer_comm_input, buffer_comm_output, done]() {
      delete input_bytes_vec;
      delete n_output;
      delete buffer_input;
      delete buffer_output;
      delete buffer_comm_input;
      delete buffer_comm_output;
      done();
    };

    OpInputList n_input_list;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("n_input", &n_input_list), done_);
    int64 total_bytes = 0;
    for (int idx = 0; idx < N_; ++idx) {
      const auto input_bytes = n_input_list[idx].TotalBytes();
      total_bytes += input_bytes;
      input_bytes_vec->at(idx) = input_bytes;
    }
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_temp(DT_INT8, {total_bytes}, buffer_input), done_);
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_temp(DT_INT8, {total_bytes}, buffer_output), done_);
    int64 offset_bytes = 0;
    for (int64 idx = 0; idx < N_; ++idx) {
      OP_REQUIRES_OK_ASYNC(ctx,
                           ctx->allocate_output(idx, n_input_list[idx].shape(),
                                                &(n_output->at(idx))),
                           done_);

      se::DeviceMemoryBase dst_ptr(
          const_cast<char*>(buffer_input->tensor_data().data()) + offset_bytes,
          input_bytes_vec->at(idx));
      ctx->op_device_context()->stream()->ThenMemcpy(
          &dst_ptr,
          se::DeviceMemoryBase(
              const_cast<char*>(n_input_list[idx].tensor_data().data()),
              input_bytes_vec->at(idx)),
          input_bytes_vec->at(idx));
      offset_bytes += input_bytes_vec->at(idx);
    }

    coll->stream()->LaunchUntilComputeDone(
        ctx, [input_bytes_vec, n_output, buffer_input, buffer_output,
              buffer_comm_input, buffer_comm_output, ctx, coll, this, done_]() {
          auto call = functor::NcclAlltoallCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              call(*buffer_input, buffer_output, buffer_comm_input,
                   buffer_comm_output, topology_, ctx, coll, this),
              done_);
          int64 offset_bytes = 0;
          for (int idx = 0; idx < N_; ++idx) {
            se::DeviceMemoryBase dst_ptr(
                const_cast<char*>(n_output->at(idx)->tensor_data().data()),
                input_bytes_vec->at(idx));
            coll->stream()->ThenMemcpy(
                &dst_ptr,
                se::DeviceMemoryBase(
                    const_cast<char*>(buffer_output->tensor_data().data()) +
                        offset_bytes,
                    input_bytes_vec->at(idx)),
                input_bytes_vec->at(idx));
            offset_bytes += input_bytes_vec->at(idx);
          }
          coll->stream()->BlockComputeUntilDone(ctx, done_);
        });
  }

 private:
  CollectiveTopology topology_;
  int64 N_;
};

#define REGISTER_KERNEL(DTYPE, WIRE_DTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAlltoallMergedN")                  \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<DTYPE>("dtype")            \
                              .TypeConstraint<WIRE_DTYPE>("wire_dtype"), \
                          NcclAlltoallMergedNOp<DTYPE, WIRE_DTYPE>);
TF_CALL_NCCL_CAST_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

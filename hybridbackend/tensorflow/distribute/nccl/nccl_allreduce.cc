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

#include "hybridbackend/tensorflow/distribute/nccl/collective.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

REGISTER_OP("HbNcclAllreduce")
    .Output("output: dtype")
    .Input("handle: resource")
    .Input("input: dtype")
    .Attr("reduce_op: int >= 0 = 0")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .SetIsStateful()
    .Doc(R"doc(
Allreduce using a NCCL communicator.

output: A reduced tensor.
handle: Handle of a NCCL communicator.
input: A tensor to reduce.
reduce_op: Reduce ops: 0 for SUM, 1 for PROD, 2 for MAX, 3 for MIN.
)doc");

#if GOOGLE_CUDA
class NcclAllreduceOp : public NcclCollectiveAsyncOp {
 public:
  explicit NcclAllreduceOp(OpKernelConstruction* ctx)
      : NcclCollectiveAsyncOp(ctx) {
    int reduce_op;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_op", &reduce_op));
    reduce_op_ = static_cast<CollectiveReduceOp>(reduce_op);
  }

  void CollectiveComputeAsync(NcclCollective* coll, OpKernelContext* ctx,
                              DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input->shape(), &output),
                         done);

    coll->stream()->LaunchUntilComputeDone(
        ctx, [input, output, this, coll, ctx, done]() {
          VLOG(1) << coll->DebugString() << " [" << name() << "] [Allreduce] ("
                  << input->TotalBytes() << "B)";
          OP_REQUIRES_OK_ASYNC(ctx, coll->Allreduce(*input, reduce_op_, output),
                               done);
          coll->stream()->BlockComputeUntilDone(ctx, done);
        });
  }

 private:
  CollectiveReduceOp reduce_op_;
};

#define REGISTER_KERNEL(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAllreduce")             \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          NcclAllreduceOp);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

REGISTER_OP("HbNcclAllreduceN")
    .Output("n_output: N * dtype")
    .Input("handle: resource")
    .Input("n_input: N * dtype")
    .Attr("reduce_op: int >= 0 = 0")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("N: int >= 1 = 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      for (int64 n = 0; n < N; ++n) {
        c->set_output(n, c->input(1 + n));
      }
      return Status::OK();
    })
    .SetIsStateful()
    .Doc(R"doc(
Packed Allreduce using a NCCL communicator.

n_output: N reduced tensors.
handle: Handle of a NCCL communicator.
n_input: N tensors to reduce.
reduce_op: Reduce ops: 0 for SUM, 1 for PROD, 2 for MAX, 3 for MIN.
)doc");

#if GOOGLE_CUDA
class NcclAllreduceNOp : public NcclCollectiveAsyncOp {
 public:
  explicit NcclAllreduceNOp(OpKernelConstruction* ctx)
      : NcclCollectiveAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    int reduce_op;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_op", &reduce_op));
    reduce_op_ = static_cast<CollectiveReduceOp>(reduce_op);
  }

  void CollectiveComputeAsync(NcclCollective* coll, OpKernelContext* ctx,
                              DoneCallback done) override {
    OpInputList n_input_list;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("n_input", &n_input_list), done);
    std::vector<Tensor>* n_input = new std::vector<Tensor>;
    std::vector<Tensor*>* n_output = new std::vector<Tensor*>(N_, nullptr);
    auto done_ = [this, n_input, n_output, done]() {
      delete n_input;
      delete n_output;
      done();
    };

    for (int64 idx = 0; idx < N_; ++idx) {
      auto& input = n_input_list[idx];
      n_input->emplace_back(input);
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(idx, input.shape(), &(n_output->at(idx))),
          done_);
    }
    coll->stream()->LaunchUntilComputeDone(ctx, [n_input, n_output, coll, ctx,
                                                 done_, this]() {
      if (VLOG_IS_ON(1)) {
        size_t input_total_bytes = 0;
        for (size_t idx = 0; idx < n_input->size(); ++idx) {
          input_total_bytes += n_input->at(idx).TotalBytes();
        }
        VLOG(1) << coll->DebugString() << " [" << name() << "] [AllreduceN] ("
                << n_input->size() << " inputs, " << input_total_bytes << "B)";
      }
      OP_REQUIRES_OK_ASYNC(
          ctx, coll->AllreduceN(*n_input, reduce_op_, n_output), done_);
      coll->stream()->BlockComputeUntilDone(ctx, done_);
    });
  }

 private:
  int64 N_;
  CollectiveReduceOp reduce_op_;
};

#define REGISTER_KERNEL(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAllreduceN")            \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          NcclAllreduceNOp);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

REGISTER_OP("HbNcclAllreduceMergedN")
    .Output("n_output: N * dtype")
    .Input("handle: resource")
    .Input("n_input: N * dtype")
    .Attr("reduce_op: int >= 0 = 0")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("N: int >= 1 = 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      for (int64 n = 0; n < N; ++n) {
        c->set_output(n, c->input(1 + n));
      }
      return Status::OK();
    })
    .SetIsStateful()
    .Doc(R"doc(
Packed merged Allreduce using a NCCL communicator.

n_output: N reduced tensors.
handle: Handle of a NCCL communicator.
n_input: N tensors to reduce.
reduce_op: Reduce ops: 0 for SUM, 1 for PROD, 2 for MAX, 3 for MIN.
)doc");

#if GOOGLE_CUDA
class NcclAllreduceMergedNOp : public NcclCollectiveAsyncOp {
 public:
  explicit NcclAllreduceMergedNOp(OpKernelConstruction* ctx)
      : NcclCollectiveAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    int reduce_op;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_op", &reduce_op));
    reduce_op_ = static_cast<CollectiveReduceOp>(reduce_op);
  }

  void CollectiveComputeAsync(NcclCollective* coll, OpKernelContext* ctx,
                              DoneCallback done) override {
    std::vector<int64>* input_bytes_vec = new std::vector<int64>(N_, 0);
    std::vector<Tensor*>* n_output = new std::vector<Tensor*>(N_, nullptr);
    Tensor* buffer_input = new Tensor();
    Tensor* buffer_output = new Tensor();
    auto done_ = [input_bytes_vec, n_output, buffer_input, buffer_output,
                  done]() {
      delete input_bytes_vec;
      delete n_output;
      delete buffer_input;
      delete buffer_output;
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

    coll->stream()->LaunchUntilComputeDone(ctx, [input_bytes_vec, n_output,
                                                 buffer_input, buffer_output,
                                                 coll, ctx, done_, this]() {
      VLOG(1) << coll->DebugString() << " [" << name()
              << "] [AllreduceMergedN] (" << buffer_input->TotalBytes() << "B)";
      OP_REQUIRES_OK_ASYNC(
          ctx, coll->Allreduce(*buffer_input, reduce_op_, buffer_output),
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
  int64 N_;
  CollectiveReduceOp reduce_op_;
};

#define REGISTER_KERNEL(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAllreduceMergedN")      \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          NcclAllreduceMergedNOp);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

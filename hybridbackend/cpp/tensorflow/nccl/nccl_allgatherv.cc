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

#include "hybridbackend/cpp/tensorflow/nccl/nccl_comm.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

REGISTER_OP("AllgathervWithNcclComm")
    .Output("output: T")
    .Input("handle: resource")
    .Input("input: T")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input = c->input(1);
      if (!c->RankKnown(input)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      int64 rank = c->Rank(input);
      if (rank == 0) {
        rank = 1;
      }  // For Allgatherv of scalar tensors.
      std::vector<shape_inference::DimensionHandle> dims(rank);
      dims[0] = c->UnknownDim();
      for (int32 i = 1; i < rank; ++i) {
        dims[i] = c->Dim(input, i);
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
AllGatherv using a NCCL communicator.

output: A gathered tensor.
handle: Handle of a NCCL communicator.
input: A tensor to gather.
)doc");

#if GOOGLE_CUDA
class AllgathervWithNcclCommOp : public NcclCommAsyncOp {
 public:
  explicit AllgathervWithNcclCommOp(OpKernelConstruction* ctx)
      : NcclCommAsyncOp(ctx) {}

  void ComputeAsyncWithComm(NcclComm* comm, OpKernelContext* ctx,
                            DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);

    Tensor* host_sizes = new Tensor();
    AllocatorAttributes host_sizes_alloc_attrs;
    host_sizes_alloc_attrs.set_on_host(true);
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->allocate_temp(DT_INT32, TensorShape({comm->size()}), host_sizes,
                           host_sizes_alloc_attrs),
        done);

    auto done_ = [host_sizes, done]() {
      delete host_sizes;
      done();
    };
    comm->RunAsync(
        [input, host_sizes, this, comm, ctx, done_]() {
          // Allocate scratch tensors.
          Tensor host_input_sizes;
          AllocatorAttributes input_sizes_alloc_attrs;
          input_sizes_alloc_attrs.set_on_host(true);
          input_sizes_alloc_attrs.set_gpu_compatible(true);
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT32, TensorShape({}), &host_input_sizes,
                                 input_sizes_alloc_attrs),
              done_);
          host_input_sizes.scalar<int32>()() = input->NumElements();

          Tensor* input_sizes = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT32, TensorShape({}), input_sizes,
                                 ctx->output_alloc_attr(0)),
              done_);
          ThenCopyToDevice(ctx, input_sizes, host_input_sizes);

          Tensor* sizes = new Tensor();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_temp(DT_INT32, TensorShape({comm->size()}), sizes,
                                 ctx->output_alloc_attr(0)),
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

          // Now all Tensor sizes have been gathered, calculate the total size
          // and allocate the output Tensor for communication.
          int32 total_size = 0;
          bool is_same_size = true;
          for (int i = 0; i < comm->size(); ++i) {
            int32 per_elements = host_sizes->flat<int32>()(i);
            total_size += per_elements;
            if (is_same_size && per_elements != input->NumElements()) {
              is_same_size = false;
            }
          }
          TensorShape out_shape(input->shape());
          int32 sub_size = 1;
          for (int i = 1; i < out_shape.dims(); ++i) {
            sub_size *= out_shape.dim_size(i);
          }
          if (out_shape.dims() == 0) {
            out_shape.AddDim(total_size);
          } else {
            out_shape.set_dim(0, total_size / sub_size);
          }
          Tensor* output;
          OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, out_shape, &output),
                               done_);
          auto outputs_ready = ThenRecordEvent(ctx);

          comm->ThenWaitFor(outputs_ready);
          if (is_same_size) {
            VLOG(1) << comm->DebugString() << " [" << name() << "] [Allgather]";
            OP_REQUIRES_OK_ASYNC(ctx, comm->Allgather(*input, output), done_);
          } else {
            VLOG(1) << comm->DebugString() << " [" << name()
                    << "] [Allgatherv]";
            OP_REQUIRES_OK_ASYNC(
                ctx, comm->Allgatherv(*input, *host_sizes, output), done_);
          }
        },
        ctx, done_);
  }
};

#define REGISTER_KERNEL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(Name("AllgathervWithNcclComm")  \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          AllgathervWithNcclCommOp);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

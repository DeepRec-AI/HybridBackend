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

REGISTER_OP("AllreduceWithNcclComm")
    .Output("output: T")
    .Input("handle: resource")
    .Input("input: T")
    .Attr("reduce_op: int >= 0 = 0")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
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
class AllreduceWithNcclCommOp : public NcclCommAsyncOp {
 public:
  explicit AllreduceWithNcclCommOp(OpKernelConstruction* ctx)
      : NcclCommAsyncOp(ctx) {
    int reduce_op;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_op", &reduce_op));
    OP_REQUIRES(ctx, reduce_op >= 0,
                errors::InvalidArgument("reduce_op is invalid:", reduce_op));
    OP_REQUIRES_OK(ctx, ReduceOpToNcclReduceOp(reduce_op, &reduce_op_));
  }

  void ComputeAsyncWithComm(NcclComm* comm, OpKernelContext* ctx,
                            DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input->shape(), &output),
                         done);

    comm->RunAsync(
        [input, output, this, comm, ctx, done]() {
          VLOG(1) << comm->DebugString() << " [" << name() << "] [Allreduce]";
          OP_REQUIRES_OK_ASYNC(ctx, comm->Allreduce(output, *input, reduce_op_),
                               done);
        },
        ctx, done);
  }

 private:
  ncclRedOp_t reduce_op_;
};

#define REGISTER_KERNEL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(Name("AllreduceWithNcclComm")   \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          AllreduceWithNcclCommOp);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

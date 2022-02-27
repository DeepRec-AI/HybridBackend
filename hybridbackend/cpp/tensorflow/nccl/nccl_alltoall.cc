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

REGISTER_OP("AlltoallWithNcclComm")
    .Output("output: T")
    .Input("handle: resource")
    .Input("input: T")
    .Attr("wire_dtype_for_float: {half, float}")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
AllToAll using a NCCL communicator.

output: Exchanged tensor for each device.
handle: Handle of a NCCL communicator.
input: Tensor to be exchanged bettween each device.
)doc");

#if GOOGLE_CUDA
class AlltoallWithNcclCommOp : public NcclCommAsyncOp {
 public:
  explicit AlltoallWithNcclCommOp(OpKernelConstruction* ctx)
      : NcclCommAsyncOp(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("wire_dtype_for_float", &wire_dtype_for_float_));
  }

  virtual void ComputeAsyncWithComm(NcclComm* comm, OpKernelContext* ctx,
                                    DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input->shape(), &output),
                         done);

    Tensor* input_in_half = new Tensor();
    Tensor* output_in_half = new Tensor();
    auto done_ = [input_in_half, output_in_half, done]() {
      delete input_in_half;
      delete output_in_half;
      done();
    };

    comm->RunAsync(
        [input, output, input_in_half, output_in_half, this, comm, ctx,
         done_]() {
          if (input->dtype() != DT_FLOAT) {
            VLOG(1) << comm->DebugString() << " [" << name() << "] [Alltoall]";
            OP_REQUIRES_OK_ASYNC(ctx, comm->Alltoall(output, *input), done_);
            return;
          }
          switch (wire_dtype_for_float_) {
            case DT_HALF: {
              OP_REQUIRES_OK_ASYNC(
                  ctx,
                  ctx->allocate_temp(DT_HALF, input->shape(), input_in_half),
                  done_);
              OP_REQUIRES_OK_ASYNC(
                  ctx,
                  ctx->allocate_temp(DT_HALF, output->shape(), output_in_half),
                  done_);
              comm->ThenWaitFor(ThenRecordEvent(ctx));
              VLOG(1) << comm->DebugString() << " [" << name() << "] [CastIn]";
              functor::CastToFloat16<float> float2half_functor_;
              float2half_functor_(ctx, input, input_in_half, comm->stream());
              VLOG(1) << comm->DebugString() << " [" << name()
                      << "] [Alltoall]";
              OP_REQUIRES_OK_ASYNC(
                  ctx, comm->Alltoall(output_in_half, *input_in_half), done_);
              VLOG(1) << comm->DebugString() << " [" << name() << "] [CastOut]";
              functor::CastFromFloat16<float> half2float_functor_;
              half2float_functor_(ctx,
                                  const_cast<const Tensor*>(output_in_half),
                                  output, comm->stream());
              break;
            }
            default:
              VLOG(1) << comm->DebugString() << " [" << name()
                      << "] [Alltoall]";
              OP_REQUIRES_OK_ASYNC(ctx, comm->Alltoall(output, *input), done_);
          }
        },
        ctx, done_);
  }

 private:
  DataType wire_dtype_for_float_;
};

#define REGISTER_KERNEL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(Name("AlltoallWithNcclComm")    \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          AlltoallWithNcclCommOp);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

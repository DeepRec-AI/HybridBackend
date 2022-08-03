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
#include "hybridbackend/tensorflow/distribute/nccl/comm.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

REGISTER_OP("HbNcclAlltoall")
    .Output("output: dtype")
    .Input("handle: resource")
    .Input("input: dtype")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
    .Attr("wire_dtype: {" TF_OP_NCCL_WIRE_DTYPE_LIST "}")
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

namespace functor {
template <typename DTYPE, typename WIRE_DTYPE>
struct NcclAlltoallCall {
  Status operator()(const Tensor& input, Tensor* output, Tensor* comm_input,
                    Tensor* comm_output, OpKernelContext* ctx, NcclComm* comm,
                    NcclCommAsyncOp* comm_op) {
    comm->ThenWaitFor(comm_op->ThenRecordEvent(ctx));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name() << "] [Alltoall]";
    TF_RETURN_IF_ERROR(comm->Alltoall(input, output));
    return Status::OK();
  }
};

template <>
struct NcclAlltoallCall<float, Eigen::half> {
  Status operator()(const Tensor& input, Tensor* output, Tensor* comm_input,
                    Tensor* comm_output, OpKernelContext* ctx, NcclComm* comm,
                    NcclCommAsyncOp* comm_op) {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_HALF, input.shape(), comm_input));
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DT_HALF, output->shape(), comm_output));
    comm->ThenWaitFor(comm_op->ThenRecordEvent(ctx));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name() << "] [CastIn]";
    functor::Cast<float, Eigen::half>()(input, comm_input, ctx, comm->stream());
    VLOG(1) << comm->DebugString() << " [" << comm_op->name() << "] [Alltoall]";
    TF_RETURN_IF_ERROR(comm->Alltoall(*comm_input, comm_output));
    VLOG(1) << comm->DebugString() << " [" << comm_op->name() << "] [CastOut]";
    functor::Cast<Eigen::half, float>()(*output, comm_output, ctx,
                                        comm->stream());
    return Status::OK();
  }
};

}  // namespace functor

template <typename DTYPE, typename WIRE_DTYPE>
class NcclAlltoallOp : public NcclCommAsyncOp {
 public:
  explicit NcclAlltoallOp(OpKernelConstruction* ctx) : NcclCommAsyncOp(ctx) {}

  virtual void ComputeAsyncWithComm(NcclComm* comm, OpKernelContext* ctx,
                                    DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input->shape(), &output),
                         done);

    Tensor* comm_input = new Tensor();
    Tensor* comm_output = new Tensor();
    auto done_ = [comm_input, comm_output, done]() {
      delete comm_input;
      delete comm_output;
      done();
    };

    comm->RunAsync(
        "NcclAlltoall", ctx, done_,
        [input, output, comm_input, comm_output, ctx, comm, this, done_]() {
          auto call = functor::NcclAlltoallCall<DTYPE, WIRE_DTYPE>();
          OP_REQUIRES_OK_ASYNC(
              ctx,
              call(*input, output, comm_input, comm_output, ctx, comm, this),
              done_);
        });
  }
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

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

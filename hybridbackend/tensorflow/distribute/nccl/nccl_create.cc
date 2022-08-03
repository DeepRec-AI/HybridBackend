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

#include "hybridbackend/tensorflow/distribute/nccl/comm.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

REGISTER_RESOURCE_HANDLE_OP(HbNcclComm);

REGISTER_OP("HbIsNcclCommInitialized")
    .Output("is_initialized: bool")
    .Input("handle: resource")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a NCCL communicator has been initialized.

is_initialized: True if the NCCL communicator is initialized.
handle: Handle of a NCCL communicator.
)doc");

REGISTER_OP("HbCreateNcclComm")
    .Input("handle: resource")
    .Input("id: int64")
    .Attr("size: int")
    .Attr("rank: int")
    .Attr("shared_name: string")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Creates a NCCL communicator and returns a handle to it.

handle: Handle of a NCCL communicator.
id: Unique ID of the NCCL communicator.
size: Total number of ranks in the communicator.
rank: Current rank in the communicator.
shared_name: Shared name of all communicator instances.
)doc");

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbNcclCommHandleOp").Device(DEVICE_GPU),
                        ResourceHandleOp<NcclComm>);

REGISTER_KERNEL_BUILDER(Name("HbIsNcclCommInitialized")
                            .Device(DEVICE_GPU)
                            .HostMemory("is_initialized")
                            .HostMemory("handle"),
                        IsResourceInitialized<NcclComm>);

class CreateNcclCommOp : public AsyncOpKernel {
 public:
  explicit CreateNcclCommOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("size", &size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rank", &rank_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
  }

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    const Tensor* id;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("id", &id), done);
    string* nccl_id =
        new string(id->tensor_data().data(), NCCL_UNIQUE_ID_BYTES);
    NcclComm* comm = new NcclComm();
    OP_REQUIRES_OK_ASYNC(ctx, comm->Initialize(size_, rank_, shared_name_, ctx),
                         done);

    comm->RunAsync(
        "NcclCommCreate", ctx, done, [this, comm, nccl_id, ctx, done]() {
          VLOG(1) << comm->DebugString() << " [" << name() << "] [Create]";
          OP_REQUIRES_OK_ASYNC(ctx, comm->Create(*nccl_id), done);
          delete nccl_id;
          Status s = CreateResource(ctx, HandleFromInput(ctx, 0), comm);
          OP_REQUIRES_ASYNC(ctx, s.ok() || s.code() == error::ALREADY_EXISTS, s,
                            done);
        });
  }

 private:
  int size_;
  int rank_;
  string shared_name_;
};

REGISTER_KERNEL_BUILDER(
    Name("HbCreateNcclComm").Device(DEVICE_GPU).HostMemory("id"),
    CreateNcclCommOp);
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

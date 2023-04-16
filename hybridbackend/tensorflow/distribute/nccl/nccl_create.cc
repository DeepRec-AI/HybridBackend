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

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/distribute/nccl/collective.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL

REGISTER_RESOURCE_HANDLE_OP(HbNcclCollective);

REGISTER_OP("HbIsNcclCollectiveInitialized")
    .Output("is_initialized: bool")
    .Input("handle: resource")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a NCCL communicator has been initialized.

is_initialized: True if the NCCL communicator is initialized.
handle: Handle of a NCCL communicator.
)doc");

REGISTER_OP("HbCreateNcclCollective")
    .Input("handle: resource")
    .Input("id: int64")
    .Attr("world_size: int")
    .Attr("local_size: int")
    .Attr("rank: int")
    .Attr("shared_name: string")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Creates a NCCL communicator and returns a handle to it.

handle: Handle of a NCCL communicator.
id: Unique ID of the NCCL communicator.
world_size: Total number of ranks in the communicator.
local_size: Total number of ranks in a node.
rank: Current rank in the communicator.
shared_name: Shared name of all communicator instances.
)doc");

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbNcclCollectiveHandleOp").Device(DEVICE_GPU),
                        ResourceHandleOp<NcclCollective>);

REGISTER_KERNEL_BUILDER(Name("HbIsNcclCollectiveInitialized")
                            .Device(DEVICE_GPU)
                            .HostMemory("is_initialized")
                            .HostMemory("handle"),
                        IsResourceInitialized<NcclCollective>);

class CreateNcclCollectiveOp : public AsyncOpKernel {
 public:
  explicit CreateNcclCollectiveOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("world_size", &world_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("local_size", &local_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rank", &rank_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
  }

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    const Tensor* id;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("id", &id), done);
    string* nccl_id =
        new string(id->tensor_data().data(), NCCL_UNIQUE_ID_BYTES);
    NcclCollective* coll =
        new NcclCollective(shared_name_, world_size_, local_size_, rank_);
    OP_REQUIRES_OK_ASYNC(ctx, coll->Initialize(ctx), done);

    coll->stream()->LaunchUntilComputeDone(ctx, [this, coll, nccl_id, ctx,
                                                 done]() {
      VLOG(1) << "[" << ctx->step_id() << "]" << coll->DebugString() << " ["
              << name() << "] [Create]";
      OP_REQUIRES_OK_ASYNC(ctx, coll->Create(*nccl_id), done);
      coll->stream()->BlockComputeUntilDone(ctx, [ctx, done, coll, nccl_id]() {
        delete nccl_id;
        Status s = CreateResource(ctx, HandleFromInput(ctx, 0), coll);
        OP_REQUIRES_ASYNC(ctx, s.ok() || s.code() == error::ALREADY_EXISTS, s,
                          done);
        static const bool kNcclAsyncErrorHandling =
            ::hybridbackend::EnvVarGetBool("NCCL_ASYNC_ERROR_HANDLING", true);
        if (kNcclAsyncErrorHandling) {
          coll->stream()->Launch(ctx, [ctx, done, coll]() {
            static const bool kNcclAsyncErrorCheckingIntervalSecs =
                ::hybridbackend::EnvVarGetInt(
                    "NCCL_ASYNC_ERROR_CHECKING_INTERVAL_SECS", 10);
            while (true) {
              OP_REQUIRES_OK_ASYNC(ctx, coll->CheckAsyncErrors(), done);
              Env::Default()->SleepForMicroseconds(
                  kNcclAsyncErrorCheckingIntervalSecs * 1000000);
            }
          });
        }
        done();
      });
    });
  }

 private:
  int world_size_;
  int local_size_;
  int rank_;
  string shared_name_;
};

REGISTER_KERNEL_BUILDER(
    Name("HbCreateNcclCollective").Device(DEVICE_GPU).HostMemory("id"),
    CreateNcclCollectiveOp);
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

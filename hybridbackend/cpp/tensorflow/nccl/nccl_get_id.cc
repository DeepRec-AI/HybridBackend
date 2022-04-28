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

namespace {
const int64 kNcclIdElements = NCCL_UNIQUE_ID_BYTES / sizeof(int64);
}  // anonymous namespace

REGISTER_OP("GetNcclId")
    .Output("id: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(kNcclIdElements));
      return Status::OK();
    })
    .SetIsStateful()
    .Doc(R"doc(
Get ID of the NCCL communciator.

id: Unique ID of the NCCL communicator.
)doc");

#if GOOGLE_CUDA
class GetNcclIdOp : public OpKernel {
 public:
  GetNcclIdOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    static_assert(NCCL_UNIQUE_ID_BYTES % sizeof(int64) == 0, "Unexpected");
    Tensor* id;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({kNcclIdElements}), &id));
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    std::memcpy(reinterpret_cast<char*>(id->flat<int64>().data()),
                nccl_id.internal, NCCL_UNIQUE_ID_BYTES);
  }
};

REGISTER_KERNEL_BUILDER(Name("GetNcclId").Device(DEVICE_GPU).HostMemory("id"),
                        GetNcclIdOp);
#endif

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

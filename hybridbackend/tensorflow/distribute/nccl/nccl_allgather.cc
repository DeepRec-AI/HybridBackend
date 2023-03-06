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

REGISTER_OP("HbNcclAllgather")
    .Output("output: dtype")
    .Input("handle: resource")
    .Input("input: dtype")
    .Attr("dtype: {" TF_OP_NCCL_DTYPE_LIST "}")
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
      }  // For Allgather of scalar tensors.
      std::vector<shape_inference::DimensionHandle> dims(rank);
      dims[0] = c->UnknownDim();
      for (int32 i = 1; i < rank; ++i) {
        dims[i] = c->Dim(input, i);
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
AllGather using a NCCL communicator.

output: A gathered tensor.
handle: Handle of a NCCL communicator.
input: A tensor to gather.
)doc");

#if GOOGLE_CUDA
class NcclAllgatherOp : public NcclCollectiveAsyncOp {
 public:
  explicit NcclAllgatherOp(OpKernelConstruction* ctx)
      : NcclCollectiveAsyncOp(ctx) {}

  void CollectiveComputeAsync(NcclCollective* coll, OpKernelContext* ctx,
                              DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);
    // Allocate output with a N times larger buffer than input, here N is the
    // communicator size which indicates the device amounts.
    TensorShape out_shape(input->shape());
    if (out_shape.dims() == 0) {
      out_shape.AddDim(coll->world_size());
    } else {
      out_shape.set_dim(0, out_shape.dim_size(0) * coll->world_size());
    }
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, out_shape, &output),
                         done);

    coll->stream()->LaunchUntilComputeDone(
        ctx, [input, output, this, coll, ctx, done]() {
          VLOG(1) << coll->DebugString() << " [" << name() << "] [Allgather]";
          OP_REQUIRES_OK_ASYNC(ctx, coll->Allgather(*input, output), done);
          coll->stream()->BlockComputeUntilDone(ctx, done);
        });
  }
};

#define REGISTER_KERNEL(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(Name("HbNcclAllgather")             \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          NcclAllgatherOp);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

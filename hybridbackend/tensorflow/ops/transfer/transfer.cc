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
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <vector>

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.pb.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/util/device_name_utils.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/common/host_functions.h"
#include "hybridbackend/tensorflow/common/stream.h"
#include "hybridbackend/tensorflow/ops/transfer/functors.h"

namespace tensorflow {
namespace hybridbackend {

using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("HbH2DTransfer")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape);

class H2DTransferOp : public OpKernel {
 public:
  explicit H2DTransferOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};

#define REGISTER_H2D_TRANSFER_KERNEL(T)                                 \
  REGISTER_KERNEL_BUILDER(Name("HbH2DTransfer").TypeConstraint<T>("T"), \
                          H2DTransferOp);

TF_CALL_ALL_TYPES(REGISTER_H2D_TRANSFER_KERNEL);
#undef REGISTER_H2D_TRANSFER_KERNEL

REGISTER_OP("HbH2DTransferN")
    .Input("input: N * T")
    .Output("output: N * T")
    .Attr("T: {" TF_OP_TRANSFER_DTYPE_LIST "}")
    .Attr("N: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 num_columns;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_columns));
      for (int i = 0; i < num_columns; ++i) {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    });

template <class T>
class H2DTransferNOp : public AsyncOpKernel {
 public:
  explicit H2DTransferNOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx), stream_(new Stream) {}

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    OpInputList input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("input", &input), done);
    OpOutputList output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("output", &output), done);
    const size_t num_inputs = input.size();

    for (int i = 0; i < num_inputs; ++i) {
      Tensor* output_ptr = nullptr;
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(i, input[i].shape(), &output_ptr), done);
    }

    functor::TransferH2DNFunctor<T>* func =
        new functor::TransferH2DNFunctor<T>(input, output, ctx);
    if (VLOG_IS_ON(3)) {
      VLOG(3) << "H2DTransferN Op " << name() << " ("
              << DataTypeString(DataTypeToEnum<T>::value) << ") has "
              << num_inputs << " inputs, including "
              << func->num_pinned_inputs() << " tensors ("
              << func->pinned_input_bytes() << "B) on pinned host memory and "
              << func->num_unpinned_inputs() << " tensors ("
              << func->unpinned_input_bytes() << "B) on unpinned host memory";
    }
    stream_->Initialize(ctx);
    stream_->LaunchUntilComputeDone(ctx, [this, ctx, done, func]() {
      OP_REQUIRES_OK_ASYNC(ctx, func->Copy(stream_->get()), done);
      stream_->BlockComputeUntilDone(ctx, [func, done]() {
        delete func;
        done();
      });
    });
  }

 private:
  Stream* stream_;
};

#define REGISTER_H2D_TRANSFER_N_KERNEL(T)              \
  REGISTER_KERNEL_BUILDER(Name("HbH2DTransferN")       \
                              .Device(DEVICE_GPU)      \
                              .HostMemory("input")     \
                              .TypeConstraint<T>("T"), \
                          H2DTransferNOp<T>)

TF_CALL_TRANSFER_TYPES(REGISTER_H2D_TRANSFER_N_KERNEL);
#undef REGISTER_H2D_TRANSFER_N_KERNEL

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

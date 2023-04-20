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

#include <bitset>

#include <tensorflow/core/common_runtime/optimization_registry.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.pb.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/util/device_name_utils.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/common/profiler.h"
#include "hybridbackend/tensorflow/common/host_functions.h"

namespace tensorflow {
namespace hybridbackend {

using GPUDevice = Eigen::GpuDevice;

class PrefetchedTransferManager : public ResourceBase {
 public:
  static PrefetchedTransferManager* Get() {
    static PrefetchedTransferManager* singleton =
        new PrefetchedTransferManager();
    return singleton;
  }

  PrefetchedTransferManager() {}

  virtual ~PrefetchedTransferManager() {}

  void Schedule(OpKernelContext* ctx, AsyncOpKernel::DoneCallback done,
                std::function<Status()> fn) {
    std::unique_lock<std::mutex> lock(mu_);

    if (!threads_) {
      threads_.reset(new thread::ThreadPool(Env::Default(), ThreadOptions(),
                                            "prefetched_transfer_threads", 1,
                                            false /* low_latency_hint */));

      int least_priority = -1;
      OP_REQUIRES_OK(ctx, CudaErrorToStatus(cudaDeviceGetStreamPriorityRange(
                              &least_priority, nullptr)));
      OP_REQUIRES_OK(ctx,
                     CudaErrorToStatus(cudaStreamCreateWithPriority(
                         &stream_, cudaStreamNonBlocking, least_priority)));
    }

    int device_id;
    cudaGetDevice(&device_id);
    auto tensor_se_stream = CudaStream(ctx->op_device_context()->stream());
    auto tensor_stream = *(tensor_se_stream.get());
    cudaEvent_t inputs_ready;
    OP_REQUIRES_OK_ASYNC(ctx,
                         CudaErrorToStatus(cudaEventCreate(
                             &inputs_ready, cudaEventDisableTiming)),
                         done);
    OP_REQUIRES_OK_ASYNC(
        ctx, CudaErrorToStatus(cudaEventRecord(inputs_ready, tensor_stream)),
        done);

    lock.unlock();
    threads_->Schedule([device_id, inputs_ready, tensor_stream, fn, this, ctx,
                        done]() {
      cudaSetDevice(device_id);
      OP_REQUIRES_OK_ASYNC(
          ctx,
          CudaErrorToStatus(cudaStreamWaitEvent(stream_, inputs_ready, 0x00)),
          done);
      OP_REQUIRES_OK_ASYNC(
          ctx, CudaErrorToStatus(cudaEventDestroy(inputs_ready)), done);

      ctx->SetStatus(fn());

      cudaEvent_t* outputs_ready = new cudaEvent_t;
      OP_REQUIRES_OK_ASYNC(ctx,
                           CudaErrorToStatus(cudaEventCreate(
                               outputs_ready, cudaEventDisableTiming)),
                           done);
      OP_REQUIRES_OK_ASYNC(
          ctx, CudaErrorToStatus(cudaEventRecord(*outputs_ready, stream_)),
          done);
      OP_REQUIRES_OK_ASYNC(ctx,
                           CudaErrorToStatus(cudaStreamWaitEvent(
                               tensor_stream, *outputs_ready, 0x00)),
                           done);

      ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
          ctx->op_device_context()->stream(),
          [ctx, tensor_stream, outputs_ready, done]() {
            OP_REQUIRES_OK_ASYNC(
                ctx, CudaErrorToStatus(cudaEventDestroy(*outputs_ready)), done);
            delete outputs_ready;
            done();
          });
    });
  }

  Status HostToDeviceCopy(const Tensor& input, Tensor* output) {
    std::unique_lock<std::mutex> lock(mu_);
    const char* h_input = input.tensor_data().data();
    char* d_output = const_cast<char*>(output->tensor_data().data());
    TF_RETURN_IF_ERROR(
        CudaErrorToStatus(cudaMemcpyAsync(d_output, h_input, input.TotalBytes(),
                                          cudaMemcpyHostToDevice, stream_)));
    return Status::OK();
  }

  Status HostToDeviceCopyN(const std::vector<Tensor>& inputs,
                           std::vector<Tensor*>* outputs) {
    std::unique_lock<std::mutex> lock(mu_);

    for (size_t idx = 0; idx < inputs.size(); ++idx) {
      const char* h_input = inputs[idx].tensor_data().data();
      char* d_output =
          const_cast<char*>(outputs->at(idx)->tensor_data().data());
      TF_RETURN_IF_ERROR(CudaErrorToStatus(
          cudaMemcpyAsync(d_output, h_input, inputs[idx].TotalBytes(),
                          cudaMemcpyHostToDevice, stream_)));
    }
    return Status::OK();
  }

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1015L
  string DebugString() const override { return "PrefetchedTransferManager()"; }
#else
  string DebugString() override { return "PrefetchedTransferManager()"; }
#endif

 private:
  std::mutex mu_;
  std::shared_ptr<thread::ThreadPool> threads_;
  cudaStream_t stream_;

  TF_DISALLOW_COPY_AND_ASSIGN(PrefetchedTransferManager);
};

REGISTER_OP("HbH2DPrefetchedTransfer")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape);

class H2DPrefetchedTransferOp : public AsyncOpKernel {
 public:
  explicit H2DPrefetchedTransferOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {}

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    Tensor* output_ptr = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, ctx->input(0).shape(), &output_ptr), done);
    PrefetchedTransferManager::Get()->Schedule(ctx, done, [ctx, output_ptr]() {
      return PrefetchedTransferManager::Get()->HostToDeviceCopy(ctx->input(0),
                                                                output_ptr);
    });
  }
};

#define REGISTER_H2D_TRANSFER_KERNEL(T)                   \
  REGISTER_KERNEL_BUILDER(Name("HbH2DPrefetchedTransfer") \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("input")        \
                              .TypeConstraint<T>("T"),    \
                          H2DPrefetchedTransferOp);

TF_CALL_ALL_TYPES(REGISTER_H2D_TRANSFER_KERNEL);
#undef REGISTER_H2D_TRANSFER_KERNEL

REGISTER_OP("HbH2DPrefetchedTransferN")
    .Input("input: N * T")
    .Output("output: N * T")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 num_columns;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_columns));
      for (int i = 0; i < num_columns; ++i) {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    });

class H2DPrefetchedTransferNOp : public AsyncOpKernel {
 public:
  explicit H2DPrefetchedTransferNOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {}

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    OpInputList input;
    OpOutputList output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("input", &input), done);
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("output", &output), done);
    OP_REQUIRES_ASYNC(
        ctx, input.size() == output.size(),
        errors::InvalidArgument("Input and output counts must match"), done);
    const size_t num_inputs = input.size();
    std::vector<Tensor>* inputs = new std::vector<Tensor>;
    std::vector<Tensor*>* output_ptrs = new std::vector<Tensor*>;
    for (int i = 0; i < num_inputs; ++i) {
      inputs->push_back(input[i]);
      Tensor* output_ptr = nullptr;
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(i, input[i].shape(), &output_ptr), done);
      output_ptrs->push_back(output_ptr);
    }

    PrefetchedTransferManager::Get()->Schedule(
        ctx, done, [ctx, inputs, output_ptrs]() {
          auto s = PrefetchedTransferManager::Get()->HostToDeviceCopyN(
              *inputs, output_ptrs);
          delete inputs;
          delete output_ptrs;
          return s;
        });
  }
};

#define REGISTER_H2D_TRANSFER_N_KERNEL(T)                  \
  REGISTER_KERNEL_BUILDER(Name("HbH2DPrefetchedTransferN") \
                              .Device(DEVICE_GPU)          \
                              .HostMemory("input")         \
                              .TypeConstraint<T>("T"),     \
                          H2DPrefetchedTransferNOp)

TF_CALL_ALL_TYPES(REGISTER_H2D_TRANSFER_N_KERNEL);
#undef REGISTER_H2D_TRANSFER_N_KERNEL

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

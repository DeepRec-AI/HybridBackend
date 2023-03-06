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

#include "hybridbackend/tensorflow/common/stream.h"
#include "hybridbackend/common/env.h"

namespace tensorflow {
namespace hybridbackend {

void Stream::Initialize(OpKernelContext* ctx) { return Initialize(ctx, "", 0); }

void Stream::Initialize(OpKernelContext* ctx, const string& name,
                        const int64 num_threads) {
  std::unique_lock<std::mutex> lock(mu_);
  if (TF_PREDICT_FALSE(se_stream_ != nullptr)) {
    return;
  }

  se_stream_ = new se::Stream(ctx->op_device_context()->stream()->parent());
  se_stream_->Init();

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1010L
  stream_ = reinterpret_cast<cudaStream_t*>(
      se_stream_->implementation()->CudaStreamMemberHack());
#else
  stream_ = reinterpret_cast<cudaStream_t*>(
      se_stream_->implementation()->GpuStreamMemberHack());
#endif

  if (num_threads == 0) {
    return;
  }

  string threads_name;
  for (size_t i = 0; i < name.size(); ++i) {
    const char ch = name[i];
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
      threads_name += ch;
    } else {
      threads_name += '_';
    }
  }
  threads_.reset(new thread::ThreadPool(ctx->env(), ThreadOptions(),
                                        threads_name, num_threads,
                                        false /* low_latency_hint */));

  return;
}

void Stream::Launch(OpKernelContext* ctx, std::function<void()> fn) {
  if (!threads_) {
    se::cuda::ScopedActivateExecutorContext context(se_stream_->parent());
    fn();
    return;
  }

  int device_id;
  cudaGetDevice(&device_id);
  threads_->Schedule([device_id, this, fn]() {
    cudaSetDevice(device_id);
    se::cuda::ScopedActivateExecutorContext context(se_stream_->parent());
    fn();
  });
}

void Stream::LaunchUntilComputeDone(OpKernelContext* ctx,
                                    std::function<void()> fn) {
  se::Event* compute_done =
      new se::Event(ctx->op_device_context()->stream()->parent());
  compute_done->Init();
  ctx->op_device_context()->stream()->ThenRecordEvent(compute_done);

  if (!threads_) {
    se_stream_->ThenWaitFor(compute_done);

    se::cuda::ScopedActivateExecutorContext context(se_stream_->parent());
    ThenExecute(ctx, [compute_done]() { delete compute_done; });
    fn();
    return;
  }

  int device_id;
  cudaGetDevice(&device_id);
  threads_->Schedule([ctx, device_id, compute_done, fn, this]() {
    cudaSetDevice(device_id);
    se_stream_->ThenWaitFor(compute_done);
    se::cuda::ScopedActivateExecutorContext context(se_stream_->parent());
    ThenExecute(ctx, [compute_done]() { delete compute_done; });
    fn();
  });
}

void Stream::BlockComputeUntilDone(OpKernelContext* ctx) {
  se::Event* ev = new se::Event(se_stream_->parent());
  ev->Init();
  se_stream_->ThenRecordEvent(ev);
  ctx->op_device_context()->stream()->ThenWaitFor(ev);
  ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
      ctx->op_device_context()->stream(), [ev]() { delete ev; });
}

void Stream::BlockComputeUntilDone(OpKernelContext* ctx,
                                   std::function<void()> fn) {
  se::Event* ev = new se::Event(se_stream_->parent());
  ev->Init();
  se_stream_->ThenRecordEvent(ev);
  ctx->op_device_context()->stream()->ThenWaitFor(ev);
  ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
      ctx->op_device_context()->stream(), [ev, fn]() {
        delete ev;
        fn();
      });
}

void Stream::BlockHostUntilDone() { se_stream_->BlockHostUntilDone(); }

Stream& Stream::ThenWaitUntilComputeDone(OpKernelContext* ctx) {
  se::Event* ev = new se::Event(ctx->op_device_context()->stream()->parent());
  ev->Init();
  ctx->op_device_context()->stream()->ThenRecordEvent(ev);
  se_stream_->ThenWaitFor(ev);
  ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
      se_stream_, [ev]() { delete ev; });
  return *this;
}

Stream& Stream::ThenExecute(OpKernelContext* ctx, std::function<void()> fn) {
  ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
      se_stream_, fn);
  return *this;
}

Stream& Stream::ThenMemcpy(void* dst, const se::DeviceMemoryBase& src,
                           uint64 size) {
  se_stream_->ThenMemcpy(dst, src, size);
  return *this;
}

Stream& Stream::ThenMemcpy(se::DeviceMemoryBase* dst, const void* src,
                           uint64 size) {
  se_stream_->ThenMemcpy(dst, src, size);
  return *this;
}

Stream& Stream::ThenMemcpy(se::DeviceMemoryBase* dst,
                           const se::DeviceMemoryBase& src, uint64 size) {
  se_stream_->ThenMemcpy(dst, src, size);
  return *this;
}

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

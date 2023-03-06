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

#ifndef HYBRIDBACKEND_TENSORFLOW_COMMON_STREAM_H_
#define HYBRIDBACKEND_TENSORFLOW_COMMON_STREAM_H_

#if HYBRIDBACKEND_TENSORFLOW
#if GOOGLE_CUDA

#include <absl/strings/str_cat.h>

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/public/version.h>

#include "hybridbackend/tensorflow/common/host_functions.h"

namespace tensorflow {
namespace hybridbackend {

class Stream {
 public:
  Stream() : se_stream_(nullptr), stream_(nullptr) {}
  virtual ~Stream() {}
  cudaStream_t* get() const { return stream_; }

  static se::Event* TensorStreamCreateEvent(OpKernelContext* ctx);

  void Initialize(OpKernelContext* ctx);
  void Initialize(OpKernelContext* ctx, const string& name,
                  const int64 num_threads);
  void Launch(OpKernelContext* ctx, std::function<void()> fn);
  void LaunchUntilComputeDone(OpKernelContext* ctx, std::function<void()> fn);

  void BlockComputeUntilDone(OpKernelContext* ctx);
  void BlockComputeUntilDone(OpKernelContext* ctx, std::function<void()> fn);
  void BlockHostUntilDone();

  Stream& ThenWaitUntilComputeDone(OpKernelContext* ctx);
  Stream& ThenExecute(OpKernelContext* ctx, std::function<void()> fn);
  Stream& ThenMemcpy(void* dst, const se::DeviceMemoryBase& src, uint64 size);
  Stream& ThenMemcpy(se::DeviceMemoryBase* dst, const void* src, uint64 size);
  Stream& ThenMemcpy(se::DeviceMemoryBase* dst, const se::DeviceMemoryBase& src,
                     uint64 size);

 private:
  std::unique_ptr<thread::ThreadPool> threads_;
  se::Stream* se_stream_;
  cudaStream_t* stream_;
  std::mutex mu_;
};
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

#endif  // HYBRIDBACKEND_TENSORFLOW_COMMON_STREAM_H_

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

#ifndef HYBRIDBACKEND_TENSORFLOW_COMMON_HOST_FUNCTIONS_H_
#define HYBRIDBACKEND_TENSORFLOW_COMMON_HOST_FUNCTIONS_H_

#if HYBRIDBACKEND_TENSORFLOW
#if GOOGLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <tensorflow/core/common_runtime/gpu/gpu_event_mgr.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/platform/cuda.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/stream_executor/stream_executor.h>

namespace tensorflow {

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1008L
namespace se = ::perftools::gputools;
#endif

namespace hybridbackend {

inline Status CudaErrorToStatus(cudaError_t rc) {
  if (!TF_PREDICT_TRUE(cudaSuccess == rc)) {
    return errors::Internal(cudaGetErrorString(rc));
  }
  return Status::OK();
}

class CudaStream {
 public:
  CudaStream(se::Stream* stream) {
#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1010L
    stream_ = reinterpret_cast<cudaStream_t*>(
        stream->implementation()->CudaStreamMemberHack());
#else
    stream_ = reinterpret_cast<cudaStream_t*>(
        stream->implementation()->GpuStreamMemberHack());
#endif
  }

  CudaStream(cudaStream_t* stream) : stream_(stream) {}

  cudaStream_t* get() const { return stream_; }

  Status ThenRecordEvent(cudaEvent_t* event_ptr) const {
    TF_RETURN_IF_ERROR(CudaErrorToStatus(
        cudaEventCreateWithFlags(event_ptr, cudaEventDisableTiming)));
    TF_RETURN_IF_ERROR(
        CudaErrorToStatus(cudaEventRecord(*event_ptr, *stream_)));
    return Status::OK();
  }

  Status ThenWaitFor(const cudaEvent_t& event) const {
    TF_RETURN_IF_ERROR(
        CudaErrorToStatus(cudaStreamWaitEvent(*stream_, event, 0)));
    cudaEventDestroy(event);
    return Status::OK();
  }

  Status ThenWaitFor(CudaStream& another) const {
    if (TF_PREDICT_FALSE(stream_ == another.get())) {
      return Status::OK();
    }
    cudaEvent_t event;
    TF_RETURN_IF_ERROR(another.ThenRecordEvent(&event));
    TF_RETURN_IF_ERROR(ThenWaitFor(event));
    return Status::OK();
  }

  Status ThenMemcpy(void* dst, const void* src, size_t count,
                    cudaMemcpyKind kind) const {
    TF_RETURN_IF_ERROR(
        CudaErrorToStatus(cudaMemcpyAsync(dst, src, count, kind, *stream_)));
    return Status::OK();
  }

  Status ThenCopyToDevice(void* dst, const void* src, size_t count) const {
    TF_RETURN_IF_ERROR(ThenMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    return Status::OK();
  }

  Status ThenCopyToHost(void* dst, const void* src, size_t count) const {
    TF_RETURN_IF_ERROR(ThenMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
    return Status::OK();
  }

  Status ThenMemset(void* devPtr, int value, size_t count) const {
    TF_RETURN_IF_ERROR(
        CudaErrorToStatus(cudaMemsetAsync(devPtr, value, count, *stream_)));
    return Status::OK();
  }

  Status BlockHostUntilDone() const {
    return CudaErrorToStatus(cudaStreamSynchronize(*stream_));
  }

  Status BlockHostUntilDone(const cudaEvent_t& event) const {
    TF_RETURN_IF_ERROR(CudaErrorToStatus(cudaEventSynchronize(event)));
    cudaEventDestroy(event);
    return Status::OK();
  }

 private:
  cudaStream_t* stream_;
};

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

#endif  // HYBRIDBACKEND_TENSORFLOW_COMMON_HOST_FUNCTIONS_H_

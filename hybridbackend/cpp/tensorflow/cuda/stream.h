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

#ifndef HYBRIDBACKEND_CPP_TENSORFLOW_CUDA_STREAM_H_
#define HYBRIDBACKEND_CPP_TENSORFLOW_CUDA_STREAM_H_

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

  cudaStream_t* get() { return stream_; }

  Status RecordEvent(cudaEvent_t* event_ptr) {
    TF_RETURN_IF_ERROR(CudaErrorToStatus(
        cudaEventCreateWithFlags(event_ptr, cudaEventDisableTiming)));
    TF_RETURN_IF_ERROR(
        CudaErrorToStatus(cudaEventRecord(*event_ptr, *stream_)));
    return Status::OK();
  }

  Status Wait() { return CudaErrorToStatus(cudaStreamSynchronize(*stream_)); }

  Status Wait(const cudaEvent_t& event) {
    TF_RETURN_IF_ERROR(CudaErrorToStatus(cudaEventSynchronize(event)));
    cudaEventDestroy(event);
    return Status::OK();
  }

 private:
  cudaStream_t* stream_;
};

template <typename D>
inline void GenerateCudaLaunchConfig(int64 work_size, const D& d,
                                     int64* block_size, int64* grid_size) {
  CHECK_GT(work_size, 0);
  const int64 actual_work_size =
      std::min(static_cast<int64>(d.getNumGpuMultiProcessors() *
                                  d.maxGpuThreadsPerMultiProcessor()),
               work_size);
  *block_size = std::min(1024, d.maxGpuThreadsPerBlock());
  *grid_size = std::min(
      static_cast<int64>((actual_work_size + *block_size - 1) / *block_size),
      static_cast<int64>(d.getNumGpuMultiProcessors()));
}

template <typename D, typename SizeType>
inline void GenerateCudaLaunchConfigByHint(const SizeType hint_block_size,
                                           const SizeType hint_min_grid_size,
                                           const SizeType work_size, const D& d,
                                           SizeType* block_size,
                                           SizeType* grid_size) {
  CHECK_GT(work_size, 0);
  *block_size = hint_block_size;
  if (*block_size < d.maxGpuThreadsPerBlock()) {
    *block_size = d.maxGpuThreadsPerBlock();
  }
  *grid_size = ((work_size - 1) / *block_size) + 1;
  if (*grid_size < hint_min_grid_size) {
    *grid_size = hint_min_grid_size;
  }
}

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

#endif  // HYBRIDBACKEND_CPP_TENSORFLOW_CUDA_STREAM_H_

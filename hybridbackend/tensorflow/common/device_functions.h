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

#ifndef HYBRIDBACKEND_TENSORFLOW_COMMON_DEVICE_FUNCTIONS_H_
#define HYBRIDBACKEND_TENSORFLOW_COMMON_DEVICE_FUNCTIONS_H_

#if HYBRIDBACKEND_TENSORFLOW
#if GOOGLE_CUDA

#include <tensorflow/core/public/version.h>
#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) < 1015L
#include <tensorflow/core/util/cuda_kernel_helper.h>
#else
#include <cuda.h>

#include <tensorflow/core/util/gpu_device_functions.h>
#include <tensorflow/core/util/gpu_kernel_helper.h>
#include <tensorflow/core/util/gpu_launch_config.h>
#endif

#include "hybridbackend/tensorflow/common/host_functions.h"

namespace tensorflow {
namespace hybridbackend {

template <typename DeviceFunc, typename... Args>
inline Status CudaLaunchKernelInternal(DeviceFunc func, int grid_size,
                                       int block_size,
                                       int dynamic_shared_memory_size,
                                       cudaStream_t stream, Args... args) {
#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) < 1014L
  func<<<grid_size, block_size, dynamic_shared_memory_size, stream>>>(args...);
  return Status::OK();
#else
  return CudaLaunchKernel(func, grid_size, block_size,
                          dynamic_shared_memory_size, stream,
                          std::forward<Args>(args)...);
#endif
}

template <typename DeviceFunc, typename Device, typename... Args>
inline Status CudaLaunch(DeviceFunc func, int size,
                         size_t dynamic_shared_memory_size, Device& d,
                         cudaStream_t* stream, Args... args) {
  auto cfg = GetCudaLaunchConfig(size, d, func, dynamic_shared_memory_size, 0);
  if (stream == nullptr) {
    return CudaLaunchKernelInternal(func, cfg.block_count, cfg.thread_per_block,
                                    dynamic_shared_memory_size, d.stream(),
                                    std::forward<Args>(args)...);
  }
  return CudaLaunchKernelInternal(func, cfg.block_count, cfg.thread_per_block,
                                  dynamic_shared_memory_size, *stream,
                                  std::forward<Args>(args)...);
}

template <typename DeviceFunc, typename Device, typename... Args>
inline Status CudaLaunchSafe(DeviceFunc func, int size,
                             size_t dynamic_shared_memory_size, Device& d,
                             cudaStream_t* stream, Args... args) {
  if (TF_PREDICT_FALSE(size < 1)) {
    return Status::OK();
  }
  int min_grid_size;
  int block_size;
  TF_RETURN_IF_ERROR(CudaErrorToStatus(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, func, dynamic_shared_memory_size, 0)));
  if (TF_PREDICT_FALSE(block_size < 1)) {
    return CudaLaunch(func, size, dynamic_shared_memory_size, d, stream,
                      std::forward<Args>(args)...);
  }
  int grid_size = ((size - 1) / block_size) + 1;
  if (stream == nullptr) {
    return CudaLaunchKernelInternal(func, grid_size, block_size,
                                    dynamic_shared_memory_size, d.stream(),
                                    std::forward<Args>(args)...);
  }
  return CudaLaunchKernelInternal(func, grid_size, block_size,
                                  dynamic_shared_memory_size, *stream,
                                  std::forward<Args>(args)...);
}

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

#endif  // HYBRIDBACKEND_TENSORFLOW_COMMON_DEVICE_FUNCTIONS_H_

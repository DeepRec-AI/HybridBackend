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

#ifndef HYBRIDBACKEND_CPP_TENSORFLOW_CUDA_DEVICE_FUNCTIONS_H_
#define HYBRIDBACKEND_CPP_TENSORFLOW_CUDA_DEVICE_FUNCTIONS_H_

#if HYBRIDBACKEND_TENSORFLOW
#if GOOGLE_CUDA

#include <tensorflow/core/public/version.h>
#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) < 1015L
#include <tensorflow/core/util/cuda_kernel_helper.h>
#else
#include <cuda.h>
#include <tensorflow/core/util/gpu_device_functions.h>
#include <tensorflow/core/util/gpu_launch_config.h>
#endif

namespace tensorflow {

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) < 1015L
#else
template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> CudaAtomicAdd(T* ptr, U value) {
  return atomicAdd(ptr, value);
}
template <typename T>
__host__ __device__ inline T ldg(const T* ptr) {
  return GpuLdg(ptr);
}
#endif

namespace hybridbackend {

template <typename T, typename N>
__global__ void SetToValue(const N count, T* ptr, T value) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1 && blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (N i : CudaGridRangeX(count)) {
    ptr[i] = value;
  }
}

template <typename T, typename N>
__global__ void SetZero(const N count, T* ptr) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1 && blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (N i : CudaGridRangeX(count)) {
    ptr[i] = T(0);
  }
}

template <typename T>
__global__ void CastToFp16(const T* in, __half* out, const int64 num_elem) {
  for (int64 input_index : CudaGridRangeX(num_elem)) {
    out[input_index] = __float2half(in[input_index]);
  }
}

template <typename T>
__global__ void CastFromFp16(const __half* in, T* out, const int64 num_elem) {
  for (int64 input_index : CudaGridRangeX(num_elem)) {
    out[input_index] = __half2float(in[input_index]);
  }
}

}  // namespace hybridbackend
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

#endif  // HYBRIDBACKEND_CPP_TENSORFLOW_CUDA_DEVICE_FUNCTIONS_H_

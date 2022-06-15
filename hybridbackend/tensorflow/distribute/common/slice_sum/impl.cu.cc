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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <limits>

#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/version.h>

#include "hybridbackend/common/atomic.cu.h"
#include "hybridbackend/tensorflow/common/device_functions.h"
#include "hybridbackend/tensorflow/distribute/common/slice_sum/functors.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace hybridbackend {

namespace functor {

template <typename T, int32 N = 256>
__global__ void SliceSumKernel(const int32 num_rows, const int32 num_cols,
                               const int32 col, const T* input, T* output_total,
                               T* output) {
  for (int32 idx : CudaGridRangeX(num_rows)) {
    const T v = input[idx * num_cols + col];
    output[idx] = v;
    atomicAdd(output_total, v);
  }
}

template <typename T>
struct SliceSum<GPUDevice, T> {
  void operator()(const int32 num_rows, const int32 num_cols, const int32 col,
                  const T* input, T* output_total, T* output,
                  const Eigen::GpuDevice& d) {
    CudaLaunch(SliceSumKernel<T>, num_rows, 0, d, nullptr, num_rows, num_cols,
               col, input, output_total, output);
  }
};

template struct SliceSum<GPUDevice, int32>;
template struct SliceSum<GPUDevice, int64>;
template struct SliceSum<GPUDevice, uint32>;
template struct SliceSum<GPUDevice, uint64>;

template <typename T, int32 N = 256>
__global__ void GroupSliceSumKernel(const int32 num_rows, const int32 num_cols,
                                    const int32 col, const int32 num_inputs,
                                    const T* inputs, T* output_totals,
                                    T** outputs) {
  for (int32 idx : CudaGridRangeX(num_inputs * num_rows)) {
    const int32 s = idx / num_rows;
    const int32 sidx = idx % num_rows;
    const T v = inputs[idx * num_cols + col];
    outputs[s][sidx] = v;
    atomicAdd(output_totals + s, v);
  }
}

template <typename T>
struct SliceSumN<GPUDevice, T> {
  void operator()(const int32 num_rows, const int32 num_cols, const int32 col,
                  const int32 num_inputs, const T* inputs, T* output_totals,
                  T** outputs, const Eigen::GpuDevice& d) {
    CudaLaunch(GroupSliceSumKernel<T>, num_inputs * num_rows, 0, d, nullptr,
               num_rows, num_cols, col, num_inputs, inputs, output_totals,
               outputs);
  }
};

template struct SliceSumN<GPUDevice, int32>;
template struct SliceSumN<GPUDevice, int64>;
template struct SliceSumN<GPUDevice, uint32>;
template struct SliceSumN<GPUDevice, uint64>;

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

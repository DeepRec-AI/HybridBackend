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
#ifndef HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_COMMON_SLICE_SUM_FUNCTORS_H_
#define HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_COMMON_SLICE_SUM_FUNCTORS_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <vector>

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/version.h>

namespace tensorflow {

class OpKernelContext;

namespace hybridbackend {
namespace functor {

template <typename Device, typename T>
struct SliceSum {
  void operator()(const int32 num_rows, const int32 num_cols, const int32 col,
                  const T* input, T* output_total, T* output,
                  const Eigen::GpuDevice& d);
};

template <typename Device, typename T>
struct SliceSumN {
  void operator()(const int32 num_rows, const int32 num_cols, const int32 col,
                  const int32 num_inputs, const T* inputs, T* output_totals,
                  T** outputs, const Eigen::GpuDevice& d);
};

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW
#endif  // HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_COMMON_SLICE_SUM_FUNCTORS_H_

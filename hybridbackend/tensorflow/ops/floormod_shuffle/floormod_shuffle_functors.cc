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

#define EIGEN_USE_THREADS

#include <limits>

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/version.h>

#include "hybridbackend/tensorflow/ops/floormod_shuffle/functors.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

namespace hybridbackend {

namespace functor {

template <typename T>
struct FloormodShuffle<CPUDevice, T> {
  void operator()(const int32 num_partitions, const Tensor& input,
                  Tensor* output, Tensor* sizes, Tensor* indices,
                  OpKernelContext* ctx) {
    const int32 input_size = input.NumElements();
    const T* h_input = input.flat<T>().data();
    T* h_output = output->flat<T>().data();
    int32* h_sizes = sizes->flat<int32>().data();
    int32* h_indices = indices->flat<int32>().data();

    std::vector<int32> local_offsets(input_size, 0);
    std::vector<int32> shard_offsets(num_partitions, 0);
    for (size_t i = 0; i < input_size; ++i) {
      const T shard =
          (h_input[i] % num_partitions + num_partitions) % num_partitions;
      local_offsets[i] = shard_offsets[shard];
      shard_offsets[shard]++;
    }
    std::memcpy(h_sizes, shard_offsets.data(), num_partitions * sizeof(int32));
    for (size_t i = 1; i < num_partitions; ++i) {
      shard_offsets[i] += shard_offsets[i - 1];
    }
    for (size_t i = 0; i < input_size; ++i) {
      const T v = h_input[i];
      const T shard = (v % num_partitions + num_partitions) % num_partitions;
      int32 offset = local_offsets[i];
      if (shard > 0) {
        offset += shard_offsets[shard - 1];
      }
      h_output[offset] = v;
      h_indices[i] = offset;
    }
  }
};

template struct FloormodShuffle<CPUDevice, int32>;
template struct FloormodShuffle<CPUDevice, int64>;
template struct FloormodShuffle<CPUDevice, uint32>;
template struct FloormodShuffle<CPUDevice, uint64>;

template <typename T>
struct FloormodShuffleN<CPUDevice, T> {
  void operator()(const int32 num_partitions, const std::vector<Tensor>& inputs,
                  std::vector<Tensor*>& outputs,
                  std::vector<Tensor*>& outputs_sizes,
                  std::vector<Tensor*>& outputs_indices, OpKernelContext* ctx) {
    OP_REQUIRES_OK(
        ctx, errors::Unimplemented("FloormodShuffleN on CPU not implemented."));
  }
};

template struct FloormodShuffleN<CPUDevice, int32>;
template struct FloormodShuffleN<CPUDevice, int64>;
template struct FloormodShuffleN<CPUDevice, uint32>;
template struct FloormodShuffleN<CPUDevice, uint64>;

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

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

#include "hybridbackend/tensorflow/distribute/partition/dual_modulo_functors.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

namespace hybridbackend {

namespace functor {

struct ComputeShardAtStageOne {
  template <typename T>
  T operator()(const T input, const int32 num_partitions, const int32 modulus) {
    return (input % num_partitions + num_partitions) % num_partitions;
  }
};

struct ComputeShardAtStageTwo {
  template <typename T>
  T operator()(const T input, const int32 num_partitions, const int32 modulus) {
    return input / modulus;
  }
};

template <typename T, typename ComputeShard>
struct PartitionByDualModulo<CPUDevice, T, ComputeShard> {
  void operator()(const int32 num_partitions, const int32 modulus,
                  const Tensor& input, Tensor* output, Tensor* sizes,
                  Tensor* indices, OpKernelContext* ctx) {
    const int32 input_size = input.NumElements();
    const T* h_input = input.flat<T>().data();
    T* h_output = output->flat<T>().data();
    int32* h_sizes = sizes->flat<int32>().data();
    int32* h_indices = indices->flat<int32>().data();

    std::vector<int32> local_offsets(input_size, 0);
    std::vector<int32> shard_offsets(num_partitions, 0);
    std::vector<T> shard_idx(input_size, 0);
    const int32 pre_mod_size = num_partitions * modulus;
    ComputeShard compute_shard;
    for (size_t i = 0; i < input_size; ++i) {
      const T pre_mod_res =
          (h_input[i] % pre_mod_size + pre_mod_size) % pre_mod_size;
      shard_idx[i] = compute_shard(pre_mod_res, num_partitions, modulus);
    }
    for (size_t i = 0; i < input_size; ++i) {
      local_offsets[i] = shard_offsets[shard_idx[i]];
      shard_offsets[shard_idx[i]]++;
    }
    std::memcpy(h_sizes, shard_offsets.data(), num_partitions * sizeof(int32));
    for (size_t i = 1; i < num_partitions; ++i) {
      shard_offsets[i] += shard_offsets[i - 1];
    }
    for (size_t i = 0; i < input_size; ++i) {
      const T v = h_input[i];
      const T shard = shard_idx[i];
      int32 offset = local_offsets[i];
      if (shard > 0) {
        offset += shard_offsets[shard - 1];
      }
      h_output[offset] = v;
      h_indices[i] = offset;
    }
  }
};

template struct PartitionByDualModulo<CPUDevice, int32, ComputeShardAtStageOne>;
template struct PartitionByDualModulo<CPUDevice, int32, ComputeShardAtStageTwo>;
template struct PartitionByDualModulo<CPUDevice, int64, ComputeShardAtStageOne>;
template struct PartitionByDualModulo<CPUDevice, int64, ComputeShardAtStageTwo>;
template struct PartitionByDualModulo<CPUDevice, uint32,
                                      ComputeShardAtStageOne>;
template struct PartitionByDualModulo<CPUDevice, uint32,
                                      ComputeShardAtStageTwo>;
template struct PartitionByDualModulo<CPUDevice, uint64,
                                      ComputeShardAtStageOne>;
template struct PartitionByDualModulo<CPUDevice, uint64,
                                      ComputeShardAtStageTwo>;

template <typename T, typename ComputeShard>
struct PartitionByDualModuloN<CPUDevice, T, ComputeShard> {
  void operator()(const int32 num_partitions, const int32 modulus,
                  const std::vector<Tensor>& inputs,
                  std::vector<Tensor*>& outputs,
                  std::vector<Tensor*>& outputs_sizes,
                  std::vector<Tensor*>& outputs_indices, OpKernelContext* ctx) {
    OP_REQUIRES_OK(ctx, errors::Unimplemented(
                            "PartitionByDualModuloN on CPU not implemented."));
  }
};

template struct PartitionByDualModuloN<CPUDevice, int32,
                                       ComputeShardAtStageOne>;
template struct PartitionByDualModuloN<CPUDevice, int32,
                                       ComputeShardAtStageTwo>;
template struct PartitionByDualModuloN<CPUDevice, int64,
                                       ComputeShardAtStageOne>;
template struct PartitionByDualModuloN<CPUDevice, int64,
                                       ComputeShardAtStageTwo>;
template struct PartitionByDualModuloN<CPUDevice, uint32,
                                       ComputeShardAtStageOne>;
template struct PartitionByDualModuloN<CPUDevice, uint32,
                                       ComputeShardAtStageTwo>;
template struct PartitionByDualModuloN<CPUDevice, uint64,
                                       ComputeShardAtStageOne>;
template struct PartitionByDualModuloN<CPUDevice, uint64,
                                       ComputeShardAtStageTwo>;

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

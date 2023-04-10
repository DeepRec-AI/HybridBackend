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

#include <limits>

#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/version.h>

#include "hybridbackend/common/atomic.cu.h"
#include "hybridbackend/common/murmur3.cu.h"

#include "hybridbackend/tensorflow/common/device_functions.h"
#include "hybridbackend/tensorflow/embedding/lookup_functors.h"

namespace tensorflow {
namespace hybridbackend {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

/// Non-blocking lookup using warp-cooperative work sharing strategy.
///
/// See https://arxiv.org/abs/1710.11246 for WCWS data structures.
///
///         slab
///  |--- warp_size ---|
///  +------+----------+---------------------|
///  | slot |          |                     |
///  +------+----------+---------------------|
///  |-- warp_size x keys_cache_slab_count --|
///
/// capacity = warp_size x keys_cache_slab_count

template <typename T>
__global__ void LookupKernel(int32* d_miss_count,
                             int32* d_hit_and_miss_keys_indices,
                             T* d_hit_cache_indices_and_miss_keys,
                             const T keys_cache_slab_count,
                             const T* d_keys_cache, const int32 key_count,
                             const T* d_keys) {
  static __constant__ T kEmptyKey = std::numeric_limits<T>::min();

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int slot = static_cast<int>(threadIdx.x % warpSize);
  T key = 0;
  T slab = 0;
  bool active = false;

  // Guess a slab to lookup.
  if (idx < key_count) {
    active = true;
    key = d_keys[idx];
    slab = murmur3_hash32(key) % keys_cache_slab_count;
  }

  int slab_hits = 0;
  int slab_miss_count = 0;
  T miss_key = kEmptyKey;
  int32 miss_idx = 0;

  const unsigned all_slots = __activemask();
  unsigned active_slots = __ballot_sync(all_slots, active);
  while (0U != active_slots) {
    int next_slot = __ffs(active_slots) - 1;
    T next_key = __shfl_sync(all_slots, key, next_slot);
    int32 next_slab = __shfl_sync(all_slots, slab, next_slot);
    int32 next_idx = __shfl_sync(all_slots, idx, next_slot);
    int32 probed_slab_count = 0;

    // Probe slabs linearly.
    const unsigned prev_active_slots = active_slots;
    while (active_slots == prev_active_slots) {
      // All slabs are probed.
      if (probed_slab_count >= keys_cache_slab_count) {
        if (slot == slab_miss_count) {
          miss_key = next_key;
          miss_idx = next_idx;
        }
        if (slot == next_slot) {
          active = false;
        }
        slab_miss_count++;
        active_slots = __ballot_sync(all_slots, active);
        break;
      }

      // Find a matching slot.
      T offset = next_slab * warpSize;
      T read_key = d_keys_cache[offset + slot];
      int good_slot =
          __ffs(__ballot_sync(all_slots, (read_key == next_key))) - 1;
      if (good_slot >= 0) {
        if (slot == next_slot) {
          d_hit_and_miss_keys_indices[next_idx] = idx;
          d_hit_cache_indices_and_miss_keys[next_idx] = offset + good_slot;
          active = false;
        }
        slab_hits++;
        active_slots = __ballot_sync(all_slots, active);
        break;
      }

      // Find an empty slot.
      if (__ballot_sync(all_slots, (kEmptyKey == read_key)) != 0) {
        if (slot == slab_miss_count) {
          miss_key = next_key;
          miss_idx = next_idx;
        }
        if (slot == next_slot) {
          active = false;
        }
        slab_miss_count++;
        active_slots = __ballot_sync(all_slots, active);
        break;
      }

      probed_slab_count++;
      next_slab = (next_slab + 1) % keys_cache_slab_count;
    }
  }

  if (slot == 0) {
    atomicAdd(d_miss_count, slab_miss_count);
  }

  if (slot < slab_miss_count) {
    d_hit_and_miss_keys_indices[key_count - 1 - miss_idx] = idx;
    d_hit_cache_indices_and_miss_keys[key_count - 1 - miss_idx] = miss_key;
  }
}

template <typename T>
void LookupFunctor<T>::operator()(int32* d_miss_count,
                                  int32* d_hit_and_miss_keys_indices,
                                  T* d_hit_cache_indices_and_miss_keys,
                                  const T keys_cache_slab_count,
                                  const T* d_keys_cache, const int32 key_count,
                                  const T* d_keys, const Eigen::GpuDevice& d) {
  CudaLaunchSafe(LookupKernel<T>, key_count, 0, d, nullptr, d_miss_count,
                 d_hit_and_miss_keys_indices, d_hit_cache_indices_and_miss_keys,
                 keys_cache_slab_count, d_keys_cache, key_count, d_keys);
}

template struct LookupFunctor<int64>;

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

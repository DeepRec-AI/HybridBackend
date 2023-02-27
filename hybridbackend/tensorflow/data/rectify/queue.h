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

#ifndef HYBRIDBACKEND_TENSORFLOW_DATA_RECTIFY_QUEUE_H_
#define HYBRIDBACKEND_TENSORFLOW_DATA_RECTIFY_QUEUE_H_

#include <deque>
#include <vector>

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/random/philox_random.h>
#include <tensorflow/core/lib/random/random.h>
#include <tensorflow/core/lib/random/random_distributions.h>

namespace tensorflow {
namespace hybridbackend {

enum TensorKinds {
  kSparseTensorIndices = 0,
  kTensorOrSparseTensorValues = 1,
  kSparseTensorDenseShape = 2
};

struct RectifyQueueItem {
 public:
  RectifyQueueItem(int64 batch_size, const std::vector<Tensor>& components)
      : batch_size(batch_size), components(components) {}
  int64 batch_size;
  std::vector<Tensor> components;
};

class RectifyQueue {
 public:
  RectifyQueue(const int64 shuffle_buffer_size, const int64 shuffle_seed,
               const int64 shuffle_seed2, const bool reshuffle_each_iteration,
               const DataTypeVector& output_dtypes,
               const std::vector<PartialTensorShape>& output_shapes,
               const std::vector<int>& output_kinds);

  int64 size() const { return size_; }

  Status Push(const int64 input_batch_size,
              const std::vector<Tensor>& input_tensors);

  Status Pop(const int64 output_batch_size, std::vector<Tensor>* output_tensors,
             Allocator* alloc);

 private:
  const int64 shuffle_buffer_size_;
  const DataTypeVector& output_dtypes_;
  const std::vector<PartialTensorShape>& output_shapes_;
  const std::vector<int>& output_kinds_;
  int64 shuffle_seed_;
  int64 shuffle_seed2_;
  int64 size_;
  std::deque<RectifyQueueItem> queue_;
  random::PhiloxRandom parent_generator_;
  random::SingleSampleAdapter<random::PhiloxRandom> generator_;
};

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW_DATA_RECTIFY_QUEUE_H_

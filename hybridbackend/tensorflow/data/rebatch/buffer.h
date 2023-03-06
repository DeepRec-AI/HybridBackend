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

#ifndef HYBRIDBACKEND_TENSORFLOW_DATA_REBATCH_BUFFER_H_
#define HYBRIDBACKEND_TENSORFLOW_DATA_REBATCH_BUFFER_H_

#include <deque>
#include <vector>

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/random/philox_random.h>
#include <tensorflow/core/lib/random/random.h>
#include <tensorflow/core/lib/random/random_distributions.h>

namespace tensorflow {
namespace hybridbackend {

struct RebatchBufferItem {
 public:
  RebatchBufferItem(int64 batch_size, const std::vector<Tensor>& components)
      : batch_size(batch_size), components(components) {}
  int64 batch_size;
  std::vector<Tensor> components;
};

class RebatchBuffer {
 public:
  RebatchBuffer(const DataTypeVector& output_dtypes,
                const std::vector<PartialTensorShape>& output_shapes,
                const std::vector<int32>& field_ranks);

  int64 size() const { return size_; }

  Status Put(const std::vector<Tensor>& input_tensors, const int64 num_rows);

  Status PutSlice(const std::vector<Tensor>& input_tensors,
                  const int64 row_start, const int64 row_limit);

  Status Shuffle(random::SingleSampleAdapter<random::PhiloxRandom>& generator,
                 const int64 num_rows);

  Status Take(Allocator* alloc, std::vector<Tensor>* output_tensors,
              const int64 num_rows);

 private:
  Status TakeDense(Allocator* alloc, std::vector<Tensor>* output_tensors,
                   std::vector<Tensor>* residual_tensors, const int64 num_rows,
                   const int64 remained_rows, const int64 rank,
                   const int64 col);

  Status TakeSparse(Allocator* alloc, std::vector<Tensor>* output_tensors,
                    std::vector<Tensor>* residual_tensors, const int64 num_rows,
                    const int64 remained_rows, const int64 rank,
                    const int64 col);

  const DataTypeVector& output_dtypes_;
  const std::vector<PartialTensorShape>& output_shapes_;
  const std::vector<int32> field_ranks_;

  int64 size_;
  std::deque<RebatchBufferItem> items_;
};

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW_DATA_REBATCH_BUFFER_H_

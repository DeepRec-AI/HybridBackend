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

#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/public/version.h>

#include <algorithm>
#include <deque>
#include <vector>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/data/rectify/queue.h"

namespace tensorflow {
namespace hybridbackend {

namespace {
template <typename T>
Status CopyValues(T* dest, const T* src, int64 num_values,
                  bool /* can_move */) {
  static_assert(is_simple_type<T>::value, "Memcpy requires a simple type.");
  memcpy(dest, src, num_values * sizeof(T));
  return Status::OK();
}

template <>
Status CopyValues<tstring>(tstring* dest, const tstring* src, int64 num_values,
                           bool can_move) {
  if (can_move) {
    for (int64 i = 0; i < num_values; ++i) {
      *dest++ = std::move(*src++);
    }
  } else {
    std::copy_n(src, num_values, dest);
  }
  return Status::OK();
}

template <>
Status CopyValues<Variant>(Variant* dest, const Variant* src, int64 num_values,
                           bool can_move) {
  if (can_move) {
    for (int64 i = 0; i < num_values; ++i) {
      *dest++ = std::move(*src++);
    }
  } else {
    std::copy_n(src, num_values, dest);
  }
  return Status::OK();
}

template <>
Status CopyValues<ResourceHandle>(ResourceHandle* dest,
                                  const ResourceHandle* src, int64 num_values,
                                  bool /* can_move */) {
  std::copy_n(src, num_values, dest);
  return Status::OK();
}

template <>
Status CopyValues<Eigen::half>(Eigen::half* dest, const Eigen::half* src,
                               int64 num_values, bool /* can_move */) {
  std::copy_n(src, num_values, dest);
  return Status::OK();
}

Status CopyToSlicesFromTensor(Tensor* dest, const int64 dest_index,
                              const Tensor& src, const bool can_move) {
  // can_move can be set by `src.RefCountIsOne();` if this function is a friend
  // of `Tensor` class
  if (src.dim_size(0) < 1) {
    return Status::OK();
  }
  DCHECK_NE(dest->dim_size(0), 0);
  DCHECK_NE(src.dim_size(0), 0);
  DCHECK_GE(dest_index, 0);
  const int64 num_values_per_record = dest->NumElements() / dest->dim_size(0);
  if (TF_PREDICT_FALSE(num_values_per_record !=
                       (src.NumElements() / src.dim_size(0)))) {
    return errors::Internal(
        "Cannot perform copy: shape of tensors does not match. "
        " Shapes are: [src]: ",
        src.shape().DebugString(), ", [dest]: ", dest->shape().DebugString());
  }
  if (TF_PREDICT_FALSE(dest_index + src.dim_size(0) > dest->dim_size(0))) {
    return errors::Internal(
        "Cannot perform copy: shape of dest tensor is too small. "
        " Shapes are: [src]: ",
        src.shape().DebugString(), ", [dest]: ", dest->shape().DebugString(),
        " Destination index is ", dest_index);
  }

#define HANDLE_TYPE(T)                                                    \
  case DataTypeToEnum<T>::value: {                                        \
    const T* src_ptr = src.unaligned_flat<T>().data();                    \
    T* dest_ptr = dest->unaligned_flat<T>().data() +                      \
                  (num_values_per_record * dest_index);                   \
    return CopyValues<T>(dest_ptr, src_ptr, src.NumElements(), can_move); \
  }

  switch (src.dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
    TF_CALL_uint32(HANDLE_TYPE);
    TF_CALL_uint64(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented(
          "CopyToSlicesFromTensor Unhandled data type: ", src.dtype());
  }
}
}  // namespace

RectifyQueue::RectifyQueue(const int64 shuffle_buffer_size,
                           const int64 shuffle_seed, const int64 shuffle_seed2,
                           const bool reshuffle_each_iteration,
                           const DataTypeVector& output_dtypes,
                           const std::vector<PartialTensorShape>& output_shapes,
                           const std::vector<int>& output_kinds)
    : shuffle_buffer_size_(shuffle_buffer_size),
      output_dtypes_(output_dtypes),
      output_shapes_(output_shapes),
      output_kinds_(output_kinds),
      shuffle_seed_(shuffle_seed),
      shuffle_seed2_(shuffle_seed2),
      size_(0),
      parent_generator_(shuffle_seed, shuffle_seed2),
      generator_(&parent_generator_) {
  if (reshuffle_each_iteration) {
    shuffle_seed_ = generator_();
    shuffle_seed2_ = generator_();
  }
}

Status RectifyQueue::Push(const int64 input_batch_size,
                          const std::vector<Tensor>& input_tensors) {
  if (shuffle_buffer_size_ > 0) {
    for (int64 row = 0; row < input_batch_size; ++row) {
      std::vector<Tensor> partial_input_tensors;
      partial_input_tensors.reserve(output_dtypes_.size());
      for (int64 col = 0; col < output_dtypes_.size(); ++col) {
        if (output_kinds_[col] == kSparseTensorDenseShape) {
          auto input_shape = input_tensors[col];
          input_shape.flat<int64>()(0) = 1;
          partial_input_tensors.push_back(std::move(input_shape));
        } else {
          partial_input_tensors.push_back(
              std::move(input_tensors[col].Slice(row, row + 1)));
        }
      }
      queue_.emplace_back(1, std::move(partial_input_tensors));
    }

    size_ += input_batch_size;
    return Status::OK();
  }

  queue_.emplace_back(input_batch_size, input_tensors);
  size_ += input_batch_size;
  return Status::OK();
}

Status RectifyQueue::Pop(const int64 output_batch_size,
                         std::vector<Tensor>* output_tensors,
                         Allocator* alloc) {
  const size_t num_components = output_dtypes_.size();
  output_tensors->clear();
  output_tensors->reserve(num_components);
  for (size_t i = 0; i < num_components; ++i) {
    PartialTensorShape output_pshape(output_shapes_[i]);
    auto kind = output_kinds_[i];
    if (kind != kSparseTensorDenseShape) {
      output_pshape.set_dim(0, output_batch_size);
    }
    TensorShape output_shape;
    output_pshape.AsTensorShape(&output_shape);
    output_tensors->emplace_back(alloc, output_dtypes_[i], output_shape);
    if (!output_tensors->back().IsInitialized()) {
      return errors::ResourceExhausted(
          "Failed to allocate memory for the batch of component ", i);
    }
  }

  if (queue_.size() < 1) {
    return Status::OK();
  }

  if (shuffle_buffer_size_ > 0) {
    for (int64 row = 0; row < output_batch_size; ++row) {
      int64 picked = row + generator_() % (queue_.size() - row);
      std::swap(queue_[picked], queue_[row]);
    }
  }

  int64 output_index = 0;
  while (output_index < output_batch_size - queue_.front().batch_size) {
    auto& front_item = queue_.front();
    if (front_item.batch_size < 1) {
      queue_.pop_front();
      continue;
    }
    for (size_t i = 0; i < num_components; ++i) {
      auto kind = output_kinds_[i];
      if (kind == kTensorOrSparseTensorValues) {
        TF_RETURN_IF_ERROR(
            CopyToSlicesFromTensor(&(output_tensors->at(i)), output_index,
                                   std::move(front_item.components[i]), true));
      } else if (kind == kSparseTensorIndices) {
        auto input = std::move(front_item.components[i]);
        int64* input_ptr = input.unaligned_flat<int64>().data();
        const int64 num_rows = input.shape().dim_size(0);
        const int64 row_elements = input.shape().num_elements() / num_rows;
        for (int64 row = 0; row < num_rows; ++row) {
          input_ptr[row * row_elements] += (output_index - input_ptr[0]);
        }
        TF_RETURN_IF_ERROR(CopyToSlicesFromTensor(&(output_tensors->at(i)),
                                                  output_index, input, true));
      } else if (output_index == 0) {
        output_tensors->at(i) = std::move(front_item.components[i]);
      }
    }
    output_index += front_item.batch_size;
    queue_.pop_front();
  }

  if (output_index < output_batch_size) {
    auto& remained_item = queue_.front();
    std::vector<Tensor> remained_components;
    remained_components.reserve(num_components);
    for (size_t i = 0; i < num_components; ++i) {
      auto kind = output_kinds_[i];
      if (kind == kTensorOrSparseTensorValues) {
        auto sliced_input = remained_item.components[i].Slice(
            0, output_batch_size - output_index);
        TF_RETURN_IF_ERROR(CopyToSlicesFromTensor(
            &(output_tensors->at(i)), output_index, sliced_input, true));
        remained_components.push_back(remained_item.components[i].Slice(
            output_batch_size - output_index, remained_item.batch_size));
      } else if (kind == kSparseTensorIndices) {
        auto sliced_input = remained_item.components[i].Slice(
            0, output_batch_size - output_index);
        int64* input_ptr = sliced_input.unaligned_flat<int64>().data();
        const int64 num_rows = sliced_input.shape().dim_size(0);
        const int64 row_elements =
            sliced_input.shape().num_elements() / num_rows;
        for (int64 row = 0; row < num_rows; ++row) {
          input_ptr[row * row_elements] += (output_index - input_ptr[0]);
        }
        TF_RETURN_IF_ERROR(CopyToSlicesFromTensor(
            &(output_tensors->at(i)), output_index, sliced_input, true));
        remained_components.push_back(remained_item.components[i].Slice(
            output_batch_size - output_index, remained_item.batch_size));
      }
    }
    int64 remained_batch_size =
        remained_item.batch_size - output_batch_size + output_index;
    output_index += remained_item.batch_size;
    queue_.pop_front();
    queue_.emplace_front(remained_batch_size, remained_components);
  }

  for (size_t i = 0; i < num_components; ++i) {
    auto kind = output_kinds_[i];
    if (kind == kSparseTensorDenseShape) {
      output_tensors->at(i).flat<int64>()(0) = output_batch_size;
    }
  }

  size_ -= output_batch_size;
  return Status::OK();
}

}  // namespace hybridbackend
}  // namespace tensorflow

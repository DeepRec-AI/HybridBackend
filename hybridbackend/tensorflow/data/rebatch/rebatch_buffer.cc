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
#include <tensorflow/core/framework/tensor_description.pb.h>
#include <tensorflow/core/public/version.h>

#include <algorithm>
#include <deque>
#include <vector>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/common/arrow.h"
#include "hybridbackend/tensorflow/common/eigen.h"
#include "hybridbackend/tensorflow/data/rebatch/buffer.h"

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

RebatchBuffer::RebatchBuffer(
    const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes,
    const std::vector<int32>& field_ranks)
    : output_dtypes_(output_dtypes),
      output_shapes_(output_shapes),
      field_ranks_(field_ranks),
      size_(0) {
  const int kArrowNumThreads =
      ::hybridbackend::EnvVarGetInt("ARROW_NUM_THREADS", 16);
  takers_.reset(new thread::ThreadPool(
      Env::Default(), ThreadOptions(), "rebatch_buffer_takers",
      kArrowNumThreads, true /* low_latency_hint */));
  int32 col = 0;
  for (int64 rank : field_ranks_) {
    field_cols_.emplace_back(col);
    col += (rank + 1);
  }

  has_zerocopied_string_.reserve(output_dtypes_.size());
}

Status RebatchBuffer::CheckZeroCopiedString(
    const std::vector<Tensor>& input_tensors) {
  has_zerocopied_string_.resize(input_tensors.size());
  auto work = [&](int64 begin, int64 end) {
    for (int64 i = begin; i < end; ++i) {
      if (input_tensors[i].dtype() != DT_STRING) {
        has_zerocopied_string_[i] = false;
      } else {
        ::tensorflow::TensorDescription tensor_description;
        input_tensors[i].FillDescription(&tensor_description);
        if (tensor_description.has_allocation_description() &&
            tensor_description.allocation_description().allocator_name() ==
                "ZerocopyArrowStringTensorBuffer") {
          has_zerocopied_string_[i] = true;
        } else {
          has_zerocopied_string_[i] = false;
        }
      }
    }
  };
  const int64 cost_per_unit = 20;
  takers_->ParallelFor(input_tensors.size(), cost_per_unit, work);
  return Status::OK();
}

Status RebatchBuffer::Put(const std::vector<Tensor>& input_tensors,
                          const int64 num_rows) {
  if (TF_PREDICT_FALSE(num_rows == 0)) {
    return Status::OK();
  }
  std::vector<int64> start(output_dtypes_.size(), 0);
  std::vector<int64> limit(output_dtypes_.size(), num_rows);
  for (int i = 0; i < input_tensors.size(); ++i) {
    if (has_zerocopied_string_[i]) {
      limit[i] = input_tensors[i].NumElements();
    }
  }
  items_.emplace_back(absl::make_unique<RebatchBufferItem>(
      num_rows, start, limit, std::move(input_tensors)));
  size_ += num_rows;
  return Status::OK();
}

Status RebatchBuffer::PutSlice(const std::vector<Tensor>& input_tensors,
                               const int64 row_start, const int64 row_limit) {
  if (TF_PREDICT_FALSE(row_start == row_limit)) {
    return Status::OK();
  }

  std::vector<Tensor> sliced_input_tensors(output_dtypes_.size());
  std::vector<int64> start_pos(output_dtypes_.size());
  std::vector<int64> limit_pos(output_dtypes_.size());

  auto work = [&](int64 begin, int64 end) {
    for (int64 i = begin; i < end; ++i) {
      int rank = field_ranks_[i];
      int col = field_cols_[i];
      int64 start = row_start;
      int64 limit = row_limit;
      if (rank != 0) {
        for (size_t split_idx = 1; split_idx < rank + 1; ++split_idx) {
          auto split_slice =
              input_tensors[col + split_idx].Slice(start, (limit + 1));
          const int64 slice_limit = (limit - start);
          start = split_slice.unaligned_flat<int32>()(0);
          limit = split_slice.unaligned_flat<int32>()(slice_limit);
          sliced_input_tensors[col + split_idx] = std::move(split_slice);
          start_pos[col + split_idx] = start;
          limit_pos[col + split_idx] = limit;
        }
      }
      const auto input_shape = input_tensors[col].shape();
      int64 input_base_elems = 1;
      if (TF_PREDICT_FALSE(input_shape.dims() > 1)) {
        input_base_elems = input_shape.num_elements() / input_shape.dim_size(0);
      }
      if (!has_zerocopied_string_[col]) {
        sliced_input_tensors[col] = input_tensors[col].Slice(
            start / input_base_elems, limit / input_base_elems);
        start_pos[col] = start / input_base_elems;
        limit_pos[col] = limit / input_base_elems;
      } else {
        sliced_input_tensors[col] = input_tensors[col];
        start_pos[col] = start;
        limit_pos[col] = limit;
      }
    }
    return Status::OK();
  };
  const int64 cost_per_unit = 20;
  takers_->ParallelFor(field_ranks_.size(), cost_per_unit, work);

  const int64 num_rows = row_limit - row_start;
  items_.emplace_back(absl::make_unique<RebatchBufferItem>(
      num_rows, start_pos, limit_pos, std::move(sliced_input_tensors)));
  size_ += num_rows;
  return Status::OK();
}

Status RebatchBuffer::Shuffle(
    random::SingleSampleAdapter<random::PhiloxRandom>& generator,
    const int64 num_rows) {
  if (size_ < num_rows) {
    return errors::InvalidArgument("Not enough rows (", size_,
                                   ") in buffer to shuffle a batch (", num_rows,
                                   ")");
  }

  for (int64 row = 0; row < num_rows; ++row) {
    int64 picked = row + generator() % (size_ - row);
    std::swap(items_[picked], items_[row]);
  }
  return Status::OK();
}

Status RebatchBuffer::Take(Allocator* alloc,
                           std::vector<Tensor>* output_tensors,
                           const int64 num_rows) {
  if (TF_PREDICT_FALSE(size_ < num_rows)) {
    return errors::InvalidArgument("Not enough rows (", size_,
                                   ") in buffer to populate a batch (",
                                   num_rows, ")");
  }

  // Compute remained rows
  int64 remained_rows = 0;
  int64 num_dirty_rows = 0;
  for (int64 row = 0; num_dirty_rows < items_.size(); ++num_dirty_rows) {
    if (row + items_[num_dirty_rows]->batch_size > num_rows) {
      remained_rows = (num_rows - row);
      break;
    }
    row += items_[num_dirty_rows]->batch_size;
  }

  const size_t num_components = output_dtypes_.size();
  output_tensors->clear();
  output_tensors->resize(num_components);
  std::vector<Tensor> residual_tensors;
  residual_tensors.resize(num_components);

  auto work = [&](int64 begin, int64 end) {
    for (int64 i = begin; i < end; ++i) {
      if (field_ranks_[i] == 0) {
        TF_RETURN_IF_ERROR(TakeDense(alloc, output_tensors, &residual_tensors,
                                     num_rows, remained_rows, field_ranks_[i],
                                     field_cols_[i]));
      } else {
        TF_RETURN_IF_ERROR(TakeSparse(alloc, output_tensors, &residual_tensors,
                                      num_rows, remained_rows, field_ranks_[i],
                                      field_cols_[i]));
      }
    }
    return Status::OK();
  };
  const int64 cost_per_unit = 200;
  takers_->ParallelFor(field_ranks_.size(), cost_per_unit, work);
  items_.erase(items_.begin(), items_.begin() + num_dirty_rows);

  if (remained_rows > 0) {
    int64 residual_rows = items_.front()->batch_size - remained_rows;
    if (residual_rows > 0) {
      items_.front()->components = residual_tensors;
      items_.front()->batch_size = residual_rows;
    } else {
      items_.erase(items_.begin());
    }
  }

  size_ -= num_rows;
  return Status::OK();
}

Status RebatchBuffer::FastPath(Allocator* alloc,
                               const std::vector<Tensor>& input_tensors,
                               std::vector<Tensor>* output_tensors) {
  auto work = [&](int64 begin, int64 end) {
    for (int i = begin; i < end; ++i) {
      if (!has_zerocopied_string_[i]) {
        output_tensors->at(i) = input_tensors[i];
      } else {
        output_tensors->at(i) =
            Tensor(alloc, input_tensors[i].dtype(), input_tensors[i].shape());
        if (!output_tensors->at(i).IsInitialized()) {
          return errors::ResourceExhausted(
              "Failed to allocate memory for output component ", i);
        }
        auto output_tensor_ptr = output_tensors->at(i).vec<std::string>();
        auto input_string_buf =
            reinterpret_cast<ArrowStringTensorBuffer*>(input_tensors[i].data());
        for (int j = 0; j < input_tensors[i].NumElements(); ++j) {
          int string_size;
          auto string_data = input_string_buf->GetValue(j, &string_size);
          output_tensor_ptr(j).assign(
              reinterpret_cast<const char*>(string_data), string_size);
        }
      }
    }
    return Status::OK();
  };

  const int64 cost_per_unit = 200;
  takers_->ParallelFor(input_tensors.size(), cost_per_unit, work);

  return Status::OK();
}

Status RebatchBuffer::TakeDense(Allocator* alloc,
                                std::vector<Tensor>* output_tensors,
                                std::vector<Tensor>* residual_tensors,
                                const int64 num_rows, const int64 remained_rows,
                                const int64 rank, const int64 col) {
  // Create output
  PartialTensorShape output_pshape(output_shapes_[col]);
  output_pshape.set_dim(0, num_rows);
  TensorShape output_shape;
  output_pshape.AsTensorShape(&output_shape);
  (*output_tensors)[col] = Tensor(alloc, output_dtypes_[col], output_shape);
  if (!output_tensors->at(col).IsInitialized()) {
    return errors::ResourceExhausted(
        "Failed to allocate memory for output component ", col);
  }

  // Populate output
  for (int64 idx = 0, row = 0; idx < items_.size(); ++idx) {
    auto& item = items_[idx]->components[col];
    if (row + items_[idx]->batch_size > num_rows) {
      if (remained_rows == 0) {
        break;
      }

      if (!has_zerocopied_string_[col]) {
        auto sliced_input = item.Slice(0, remained_rows);
        TF_RETURN_IF_ERROR(CopyToSlicesFromTensor(
            &(output_tensors->at(col)), row, std::move(sliced_input), true));
        (*residual_tensors)[col] =
            item.Slice(remained_rows, items_[idx]->batch_size);
        items_[idx]->start[col] = remained_rows;
      } else {
        int64 start = items_[idx]->start[col];
        int64 limit = start + remained_rows;
        auto output_tensor_ptr = output_tensors->at(col).vec<std::string>();
        auto item_string_buf_ptr =
            reinterpret_cast<ArrowStringTensorBuffer*>(item.data());
        for (int i = start; i < limit; ++i) {
          int64 output_idx = row + i - start;
          int string_size;
          auto string_data = item_string_buf_ptr->GetValue(i, &string_size);
          output_tensor_ptr(output_idx)
              .assign(reinterpret_cast<const char*>(string_data), string_size);
        }
        (*residual_tensors)[col] = item;
        items_[idx]->start[col] = limit;
      }
      break;
    }
    if (!has_zerocopied_string_[col]) {
      TF_RETURN_IF_ERROR(
          CopyToSlicesFromTensor(&(output_tensors->at(col)), row, item, true));
    } else {
      // populating string type of output tensors
      int64 item_start_pos = items_[idx]->start[col];
      int64 item_limit_pos = items_[idx]->limit[col];
      auto output_tensor_ptr = output_tensors->at(col).vec<std::string>();
      auto item_string_buf_ptr =
          reinterpret_cast<ArrowStringTensorBuffer*>(item.data());
      for (int i = item_start_pos; i < item_limit_pos; ++i) {
        int64 output_idx = row + i - item_start_pos;
        int string_size;
        auto string_data = item_string_buf_ptr->GetValue(i, &string_size);
        output_tensor_ptr(output_idx)
            .assign(reinterpret_cast<const char*>(string_data), string_size);
      }
    }
    row += items_[idx]->batch_size;
  }

  return Status::OK();
}

Status RebatchBuffer::TakeSparse(Allocator* alloc,
                                 std::vector<Tensor>* output_tensors,
                                 std::vector<Tensor>* residual_tensors,
                                 const int64 num_rows,
                                 const int64 remained_rows, const int64 rank,
                                 const int64 col) {
  // Create and populate output splits
  int64 remained_dim0_size = 1 + remained_rows;
  for (size_t split_idx = 1; split_idx < rank + 1; ++split_idx) {
    int64 next_remained_dim0_size = 0;
    int64 dim0_size = 0;
    for (int64 idx = 0, row = 0; idx < items_.size(); ++idx) {
      if (row + items_[idx]->batch_size > num_rows) {
        next_remained_dim0_size =
            items_[idx]->components[col + split_idx].unaligned_flat<int32>()(
                remained_dim0_size - 1);
        dim0_size += (remained_dim0_size - 1);
        break;
      }
      dim0_size += (items_[idx]->components[col + split_idx].dim_size(0) - 1);
      row += items_[idx]->batch_size;
    }

    PartialTensorShape split_pshape(output_shapes_[col + split_idx]);
    split_pshape.set_dim(0, 1 + dim0_size);
    TensorShape split_shape;
    split_pshape.AsTensorShape(&split_shape);
    (*output_tensors)[col + split_idx] = Tensor(alloc, DT_INT32, split_shape);

    if (!output_tensors->at(col + split_idx).IsInitialized()) {
      return errors::ResourceExhausted(
          "Failed to allocate memory for output component ", col);
    }

    (*output_tensors)[col + split_idx].unaligned_flat<int32>()(0) = 0;
    int64 dim0_index = 0;
    for (int64 idx = 0, row = 0; idx < items_.size(); ++idx) {
      auto& split = items_[idx]->components[col + split_idx];
      int32 output_last = output_tensors->at(col + split_idx)
                              .unaligned_flat<int32>()(dim0_index);
      int32 input_first = split.unaligned_flat<int32>()(0);
      if (row + items_[idx]->batch_size > num_rows) {
        if (remained_rows == 0) {
          break;
        }
        auto sliced_input_split = split.Slice(1, remained_dim0_size);
        TF_RETURN_IF_ERROR(CopyToSlicesFromTensor(
            &(output_tensors->at(col + split_idx)), 1 + dim0_index,
            std::move(sliced_input_split), true));
        auto sliced_output_split =
            output_tensors->at(col + split_idx)
                .Slice(1 + dim0_index, remained_dim0_size + dim0_index);
        sliced_output_split.unaligned_flat<int32>() +=
            sliced_output_split.unaligned_flat<int32>().constant(output_last -
                                                                 input_first);
        auto residual_input_split =
            split.Slice(remained_dim0_size - 1, split.dim_size(0));
        (*residual_tensors)[col + split_idx] = std::move(residual_input_split);
        break;
      }
      auto sliced_input_split = split.Slice(1, split.dim_size(0));
      TF_RETURN_IF_ERROR(CopyToSlicesFromTensor(
          &(output_tensors->at(col + split_idx)), 1 + dim0_index,
          std::move(sliced_input_split), true));
      auto sliced_output_split =
          output_tensors->at(col + split_idx)
              .Slice(1 + dim0_index, split.dim_size(0) + dim0_index);
      sliced_output_split.unaligned_flat<int32>() +=
          sliced_output_split.unaligned_flat<int32>().constant(output_last -
                                                               input_first);
      dim0_index += sliced_input_split.dim_size(0);
      row += items_[idx]->batch_size;
    }

    remained_dim0_size = next_remained_dim0_size;
  }

  // Create and populate ouput values
  int64 values_dim0_size = 0;
  for (int64 idx = 0, row = 0; idx < items_.size(); ++idx) {
    int64 item_start_pos = items_[idx]->start[col];
    int64 item_limit_pos = items_[idx]->limit[col];
    if (row + items_[idx]->batch_size > num_rows) {
      break;
    }
    if (!has_zerocopied_string_[col]) {
      values_dim0_size += items_[idx]->components[col].dim_size(0);
    } else {
      values_dim0_size += (item_limit_pos - item_start_pos);
    }
    row += items_[idx]->batch_size;
  }
  PartialTensorShape base_pshape(output_shapes_[col]);
  base_pshape.set_dim(0, 1);
  if (remained_rows > 0) {
    values_dim0_size += remained_dim0_size / base_pshape.num_elements();
  }
  PartialTensorShape output_pshape(output_shapes_[col]);
  output_pshape.set_dim(0, values_dim0_size);
  TensorShape output_shape;
  output_pshape.AsTensorShape(&output_shape);
  (*output_tensors)[col] = Tensor(alloc, output_dtypes_[col], output_shape);
  if (!output_tensors->at(col).IsInitialized()) {
    return errors::ResourceExhausted(
        "Failed to allocate memory for output component ", col);
  }

  int64 dim0_index = 0;
  for (int64 idx = 0, row = 0; idx < items_.size(); ++idx) {
    auto& values = items_[idx]->components[col];
    int64 item_start_pos = items_[idx]->start[col];
    int64 item_limit_pos = items_[idx]->limit[col];

    if (row + items_[idx]->batch_size > num_rows) {
      if (remained_rows == 0) {
        break;
      }
      const auto values_shape = values.shape();
      int64 values_base_elems = 1;
      if (TF_PREDICT_FALSE(values_shape.dims() > 1)) {
        values_base_elems =
            values_shape.num_elements() / values_shape.dim_size(0);
      }
      if (!has_zerocopied_string_[col]) {
        auto sliced_values =
            values.Slice(0, remained_dim0_size / values_base_elems);
        TF_RETURN_IF_ERROR(
            CopyToSlicesFromTensor(&(output_tensors->at(col)), dim0_index,
                                   std::move(sliced_values), true));
        (*residual_tensors)[col] = values.Slice(
            remained_dim0_size / values_base_elems, values.dim_size(0));
      } else {
        int64 start = items_[idx]->start[col];
        int64 limit = start + remained_dim0_size;
        auto output_tensor_ptr = output_tensors->at(col).vec<std::string>();
        auto item_string_buf_ptr =
            reinterpret_cast<ArrowStringTensorBuffer*>(values.data());
        for (int i = start; i < limit; ++i) {
          int64 output_idx = dim0_index + i - start;
          int string_size;
          auto string_data = item_string_buf_ptr->GetValue(i, &string_size);
          output_tensor_ptr(output_idx)
              .assign(reinterpret_cast<const char*>(string_data), string_size);
        }
        (*residual_tensors)[col] = values;
        items_[idx]->start[col] = limit;
      }
      break;
    }

    if (!has_zerocopied_string_[col]) {
      TF_RETURN_IF_ERROR(CopyToSlicesFromTensor(&(output_tensors->at(col)),
                                                dim0_index, values, true));
    } else {
      auto output_tensor_ptr = output_tensors->at(col).vec<std::string>();
      auto item_string_buf_ptr =
          reinterpret_cast<ArrowStringTensorBuffer*>(values.data());
      for (int i = item_start_pos; i < item_limit_pos; ++i) {
        int64 output_idx = dim0_index + i - item_start_pos;
        int string_size;
        auto string_data = item_string_buf_ptr->GetValue(i, &string_size);
        output_tensor_ptr(output_idx)
            .assign(reinterpret_cast<const char*>(string_data), string_size);
      }
    }

    if (!has_zerocopied_string_[col]) {
      dim0_index += values.dim_size(0);
    } else {
      dim0_index += (item_limit_pos - item_start_pos);
    }
    row += items_[idx]->batch_size;
  }

  return Status::OK();
}

}  // namespace hybridbackend
}  // namespace tensorflow

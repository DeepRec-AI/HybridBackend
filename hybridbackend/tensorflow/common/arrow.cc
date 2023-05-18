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

#include <unistd.h>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#if HYBRIDBACKEND_ARROW
#include <arrow/array.h>
#include <arrow/util/thread_pool.h>
#include <tensorflow/core/framework/allocation_description.pb.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/common/arrow.h"
#include "hybridbackend/tensorflow/common/eigen.h"
#endif

namespace tensorflow {
namespace hybridbackend {

namespace {
inline bool ZeroCopyStringForRebatchDisabled() {
  static const bool kZeroCopyStringForRebatchDisabled =
      ::hybridbackend::EnvVarGetBool("HB_ZERO_COPY_STRING_FOR_REBATCH_DISABLED",
                                     false);
  return kZeroCopyStringForRebatchDisabled;
}
}  // namespace

#if HYBRIDBACKEND_ARROW

#if HYBRIDBACKEND_ARROW_ZEROCOPY
#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) < 1014L
ArrowStringTensorBuffer::ArrowStringTensorBuffer(
    const std::shared_ptr<arrow::Buffer>& value_data_buf,
    const std::shared_ptr<arrow::Buffer>& value_offsets_buf,
    const uint8_t* raw_data, const int32_t* raw_value_offsets)
    : value_data_buf_(value_data_buf),
      value_offsets_buf_(value_offsets_buf),
      raw_data_(raw_data),
      raw_value_offsets_(raw_value_offsets) {}

void ArrowStringTensorBuffer::data() const { return this; }

#else
ArrowStringTensorBuffer::ArrowStringTensorBuffer(
    const std::shared_ptr<arrow::Buffer>& value_data_buf,
    const std::shared_ptr<arrow::Buffer>& value_offsets_buf,
    const uint8_t* raw_data, const int32_t* raw_value_offsets)
    : TensorBuffer(this),
      value_data_buf_(value_data_buf),
      value_offsets_buf_(value_offsets_buf),
      raw_data_(raw_data),
      raw_value_offsets_(raw_value_offsets) {}
#endif

size_t ArrowStringTensorBuffer::size() const {
  LOG(ERROR) << "When using zero copy string for rebatch, please and a "
                "hb.data.rebatch(batch_size) following hb.data.ParquetDataset ";
  return 0;
}

TensorBuffer* ArrowStringTensorBuffer::root_buffer() { return this; }

void ArrowStringTensorBuffer::FillAllocationDescription(
    AllocationDescription* proto) const {
  proto->set_requested_bytes(sizeof(tstring));
  proto->set_allocator_name("ZerocopyArrowStringTensorBuffer");
}

bool ArrowStringTensorBuffer::OwnsMemory() const { return false; }

const uint8_t* ArrowStringTensorBuffer::GetValue(int64_t i,
                                                 int32_t* out_length) {
  const int32_t pos = raw_value_offsets_[i];
  *out_length = raw_value_offsets_[i + 1] - pos;
  return raw_data_ + pos;
}
#endif

namespace {
#if HYBRIDBACKEND_ARROW_ZEROCOPY
class ArrowPrimitiveTensorBuffer : public TensorBuffer {
 public:
  ArrowPrimitiveTensorBuffer() = delete;

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) < 1014L
  explicit ArrowPrimitiveTensorBuffer(
      const std::shared_ptr<arrow::Buffer>& arrow_buffer)
      : arrow_buffer_(arrow_buffer) {}

  void* data() const override {
    return const_cast<uint8_t*>(arrow_buffer_->data());
  }
#else
  explicit ArrowPrimitiveTensorBuffer(
      const std::shared_ptr<arrow::Buffer>& arrow_buffer)
      : TensorBuffer(const_cast<uint8_t*>(arrow_buffer->data())),
        arrow_buffer_(arrow_buffer) {}
#endif

  size_t size() const override { return arrow_buffer_->size(); }

  TensorBuffer* root_buffer() override { return this; }

  void FillAllocationDescription(AllocationDescription* proto) const override {
    proto->set_requested_bytes(size());
    proto->set_allocator_name(::tensorflow::cpu_allocator()->Name());
  }

  bool OwnsMemory() const override { return false; }

 private:
  std::shared_ptr<::arrow::Buffer> arrow_buffer_;
};
#endif

::arrow::Status MakeTensorFromArrowBuffer(
    const DataType dtype, const PartialTensorShape& shape,
    const std::shared_ptr<::arrow::Buffer>& arrow_buffer, Tensor* tensor) {
  const int64 total_num_elems = arrow_buffer->size() / DataTypeSize(dtype);
  int64 shape_num_elems = shape.num_elements();
  if (TF_PREDICT_FALSE(shape_num_elems < 0)) {
    return ::arrow::Status::Invalid(
        "Supposed shape of input batch is not fully defined");
  }
  int64 dim0 = total_num_elems;
  if (TF_PREDICT_FALSE(shape_num_elems > 0)) {
    dim0 = total_num_elems / shape_num_elems;
    if (TF_PREDICT_FALSE(dim0 * shape_num_elems != total_num_elems)) {
      return ::arrow::Status::Invalid(
          "Supposed shape and actual shape of input batch mismatches");
    }
  }
  TensorShape actual_shape;
  if (!TF_PREDICT_TRUE(
          PartialTensorShape({dim0}).Concatenate(shape).AsTensorShape(
              &actual_shape))) {
    return ::arrow::Status::Invalid(
        "Calculated shape of input batch is not fully defined");
  }

#if HYBRIDBACKEND_ARROW_ZEROCOPY
  // NOTE: Alignment is 64 in Arrow, same to EIGEN_MAX_ALIGN_BYTES. See:
  // https://github.com/apache/arrow/blob/apache-arrow-9.0.0/cpp/src/arrow/memory_pool_internal.h#L29
  if (TF_PREDICT_FALSE(!CHECK_EIGEN_ALIGN(arrow_buffer->data()))) {
    *tensor = Tensor(dtype, actual_shape);
    std::memcpy(const_cast<char*>(tensor->tensor_data().data()),
                arrow_buffer->data(), arrow_buffer->size());
    return ::arrow::Status::OK();
  }

  ArrowPrimitiveTensorBuffer* tensor_buffer =
      new ArrowPrimitiveTensorBuffer(arrow_buffer);
  core::ScopedUnref unref(tensor_buffer);
  *tensor = Tensor(dtype, actual_shape, tensor_buffer);
  return ::arrow::Status::OK();
#else
  *tensor = Tensor(dtype, actual_shape);
  std::memcpy(const_cast<char*>(tensor->tensor_data().data()),
              arrow_buffer->data(), arrow_buffer->size());
  return ::arrow::Status::OK();
#endif
}

::arrow::Status MakeStringTensorFromArrowArray(
    const PartialTensorShape& shape, const ::arrow::StringArray& array,
    Tensor* tensor) {
  if (array.null_count() != 0) {
    return arrow::Status::Invalid("Null elements not supported");
  }

  const auto total_num_elems = array.length();
  int64 shape_num_elems = shape.num_elements();
  if (TF_PREDICT_FALSE(shape_num_elems < 0)) {
    return ::arrow::Status::Invalid("Field shape is not fully defined");
  }
  int64 dim0 = total_num_elems;
  if (TF_PREDICT_FALSE(shape_num_elems > 0)) {
    dim0 = total_num_elems / shape_num_elems;
    if (TF_PREDICT_FALSE(dim0 * shape_num_elems != total_num_elems)) {
      return ::arrow::Status::Invalid("Field shape mismatch with actual data");
    }
  }
  TensorShape actual_shape;
  if (!TF_PREDICT_TRUE(
          PartialTensorShape({dim0}).Concatenate(shape).AsTensorShape(
              &actual_shape))) {
    return ::arrow::Status::Invalid("Field shape is not fully defined");
  }
  if (ZeroCopyStringForRebatchDisabled()) {
    *tensor = Tensor(DT_STRING, actual_shape);
    auto tensor_vec = tensor->vec<std::string>();

    for (auto i = 0; i < total_num_elems; ++i) {
      int string_size;
      auto string_data = array.GetValue(i, &string_size);
      tensor_vec(i).assign(reinterpret_cast<const char*>(string_data),
                           string_size);
    }
  } else {
#if HYBRIDBACKEND_ARROW_ZEROCOPY
    ArrowStringTensorBuffer* tensor_buffer = new ArrowStringTensorBuffer(
        array.value_data(), array.value_offsets(), array.raw_data(),
        array.raw_value_offsets());
    core::ScopedUnref unref(tensor_buffer);
    *tensor = Tensor(DT_STRING, actual_shape, tensor_buffer);
#else
    *tensor = Tensor(DT_STRING, actual_shape);
    auto tensor_vec = tensor->vec<std::string>();

    for (auto i = 0; i < total_num_elems; ++i) {
      int string_size;
      auto string_data = array.GetValue(i, &string_size);
      tensor_vec(i).assign(reinterpret_cast<const char*>(string_data),
                           string_size);
    }
#endif
  }
  return ::arrow::Status::OK();
}

// Primitive Arrow arrays have validity and value buffers.
#define RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(ARRAY_CLASS)                  \
  ::arrow::Status Visit(const ARRAY_CLASS& array) override {                \
    if (TF_PREDICT_FALSE(unvisited_ragged_indices_ != 0)) {                 \
      return ::arrow::Status::Invalid("Inconsistent ragged rank");          \
    }                                                                       \
    Tensor tensor;                                                          \
    auto st = MakeTensorFromArrowBuffer(dtype_, shape_,                     \
                                        array.data()->buffers[1], &tensor); \
    if (!st.ok()) {                                                         \
      return st;                                                            \
    }                                                                       \
    ragged_tensor_.push_front(std::move(tensor));                           \
    return ::arrow::Status::OK();                                           \
  }

#define RAGGED_TENSOR_BUILDER_STRING_VISIT(ARRAY_CLASS)               \
  ::arrow::Status Visit(const ARRAY_CLASS& array) override {          \
    if (TF_PREDICT_FALSE(unvisited_ragged_indices_ != 0)) {           \
      return ::arrow::Status::Invalid("Inconsistent ragged rank");    \
    }                                                                 \
    Tensor tensor;                                                    \
    auto st = MakeStringTensorFromArrowArray(shape_, array, &tensor); \
    if (!st.ok()) {                                                   \
      return st;                                                      \
    }                                                                 \
    ragged_tensor_.push_front(std::move(tensor));                     \
    return ::arrow::Status::OK();                                     \
  }

class RaggedTensorBuilder : public ::arrow::ArrayVisitor {
 public:
  RaggedTensorBuilder(const DataType dtype, const int32 ragged_rank,
                      const PartialTensorShape& shape)
      : dtype_(dtype),
        ragged_rank_(ragged_rank),
        shape_(shape),
        unvisited_ragged_indices_(ragged_rank) {}

  ::arrow::Status Build(const std::shared_ptr<::arrow::Array>& array,
                        std::vector<Tensor>* output_tensors) {
    auto st = array->Accept(this);
    if (!st.ok()) {
      return st;
    }

    // Follow RaggedTensor-style ordering: V, Sn, Sn-1, ..., S0
    if (ragged_tensor_.size() > 1) {
      std::reverse(std::next(ragged_tensor_.begin()), ragged_tensor_.end());
    }

    output_tensors->insert(output_tensors->end(), ragged_tensor_.begin(),
                           ragged_tensor_.end());
    return ::arrow::Status::OK();
  }

  ::arrow::Status Visit(const ::arrow::ListArray& array) override {
    --unvisited_ragged_indices_;
    Tensor tensor;
    auto st = MakeTensorFromArrowBuffer(DT_INT32, PartialTensorShape({}),
                                        array.value_offsets(), &tensor);
    if (!st.ok()) {
      return st;
    }
    ragged_tensor_.push_front(std::move(tensor));
    return array.values()->Accept(this);
  }

  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::Int8Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::UInt8Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::Int32Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::UInt32Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::Int64Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::UInt64Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::HalfFloatArray);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::FloatArray);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::DoubleArray);

  RAGGED_TENSOR_BUILDER_STRING_VISIT(::arrow::StringArray);

 private:
  const DataType dtype_;
  const int32 ragged_rank_;
  const PartialTensorShape shape_;
  int32 unvisited_ragged_indices_;
  std::deque<Tensor> ragged_tensor_;
};

}  // namespace

#define CASE_ARROW_ENUM_SET_DTYPE(PTR, ENUM)                       \
  case ENUM: {                                                     \
    *PTR = DataTypeToEnum<ArrowEnumToDataType<ENUM>::Type>::value; \
    return Status::OK();                                           \
  }

Status MakeDataTypeAndRaggedRankFromArrowDataType(
    const std::shared_ptr<::arrow::DataType>& arrow_dtype, DataType* dtype,
    int32* ragged_rank) {
  if (arrow_dtype->id() == ::arrow::Type::LIST) {
    ++(*ragged_rank);
    return MakeDataTypeAndRaggedRankFromArrowDataType(
        arrow_dtype->field(0)->type(), dtype, ragged_rank);
  }

  switch (arrow_dtype->id()) {
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::INT8);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::UINT8);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::INT32);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::UINT32);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::INT64);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::UINT64);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::HALF_FLOAT);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::FLOAT);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::DOUBLE);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::STRING);
    default:
      return errors::Unimplemented("Arrow data type ", arrow_dtype->ToString(),
                                   " not supported.");
  }
  return Status::OK();
}

Status MakeTensorsFromArrowArray(
    const DataType dtype, const int32 ragged_rank,
    const PartialTensorShape& shape,
    const std::shared_ptr<::arrow::Array>& arrow_array,
    std::vector<Tensor>* output_tensors) {
  if (TF_PREDICT_FALSE(arrow_array->null_count() != 0)) {
    return errors::Internal("Data with null values not supported");
  }

  if (TF_PREDICT_FALSE(arrow_array->data()->offset != 0)) {
    return errors::Internal("Data has zero non-offset not supported");
  }

  RaggedTensorBuilder builder(dtype, ragged_rank, shape);
  TF_RETURN_IF_ARROW_ERROR(builder.Build(arrow_array, output_tensors));
  return Status::OK();
}

Status ValidateSchema(const string& filename,
                      const std::vector<string>& field_names,
                      const DataTypeVector& field_dtypes,
                      const std::vector<int32>& field_ragged_ranks,
                      std::shared_ptr<::arrow::Schema>& schema,
                      std::vector<int>* out_column_indices) {
  if (TF_PREDICT_FALSE(!schema->HasDistinctFieldNames())) {
    return errors::InvalidArgument(filename, " must has distinct column names");
  }
  for (size_t i = 0; i < field_names.size(); ++i) {
    auto& cname = field_names[i];
    int column_index = schema->GetFieldIndex(cname);
    if (TF_PREDICT_FALSE(column_index < 0)) {
      return errors::NotFound("No column called `", cname, "` found in ",
                              filename);
    }
    if (out_column_indices != nullptr) {
      out_column_indices->push_back(column_index);
    }
    const auto& expected_dtype = field_dtypes[i];
    const auto& expected_ragged_rank = field_ragged_ranks[i];
    DataType actual_dtype;
    int32 actual_ragged_rank = 0;
    TF_RETURN_IF_ERROR(MakeDataTypeAndRaggedRankFromArrowDataType(
        schema->field(column_index)->type(), &actual_dtype,
        &actual_ragged_rank));
    if (TF_PREDICT_FALSE(actual_dtype != expected_dtype)) {
      return errors::InvalidArgument(
          "Field ", cname, " in ", filename, " has unexpected data type ",
          DataTypeString(actual_dtype), ", which should be ",
          DataTypeString(expected_dtype));
    }
    if (TF_PREDICT_FALSE(actual_ragged_rank != expected_ragged_rank)) {
      return errors::InvalidArgument(
          "Field ", cname, " in ", filename, " has unexpected ragged rank ",
          actual_ragged_rank, ", which should be ", expected_ragged_rank);
    }
  }
  return Status::OK();
}

Status ReadRecordBatch(::arrow::RecordBatchReader* batch_reader,
                       const string& filename, const int64 batch_size,
                       const std::vector<string>& field_names,
                       const DataTypeVector& field_dtypes,
                       const std::vector<int32>& field_ragged_ranks,
                       const std::vector<PartialTensorShape>& field_shapes,
                       const bool drop_remainder, const int64 row_limit,
                       std::vector<Tensor>* output_tensors,
                       int64* row_counter) {
#if HYBRIDBACKEND_ARROW
  // Read next batch from parquet file.
  std::shared_ptr<::arrow::RecordBatch> batch;
  TF_RETURN_IF_ARROW_ERROR(batch_reader->ReadNext(&batch));
  if (TF_PREDICT_FALSE(!batch)) {
    return errors::OutOfRange("Reached end of parquet file ", filename);
  }
  if (TF_PREDICT_FALSE(drop_remainder && batch->num_rows() < batch_size)) {
    return errors::OutOfRange("Reached end of parquet file ", filename,
                              " after dropping reminder batch");
  }

  if (TF_PREDICT_FALSE(row_limit > -1 &&
                       (*row_counter) + batch->num_rows() > row_limit)) {
    batch = batch->Slice(0, batch->num_rows() - (row_limit - (*row_counter)));
  }

  // Populate tensors from record batch.
  auto arrays = batch->columns();
  for (size_t i = 0; i < arrays.size(); ++i) {
    auto s =
        MakeTensorsFromArrowArray(field_dtypes[i], field_ragged_ranks[i],
                                  field_shapes[i], arrays[i], output_tensors);
    if (!s.ok()) {
      return errors::DataLoss("Failed to parse row #", *row_counter, " - #",
                              (*row_counter) + batch->num_rows(), " at column ",
                              field_names[i], " (#", i, ") in ", filename, ": ",
                              s.error_message());
    }
  }

  (*row_counter) += batch->num_rows();
  return Status::OK();
#else
  return errors::Unimplemented("HYBRIDBACKEND_WITH_ARROW must be ON");
#endif
}

#endif

}  // namespace hybridbackend
}  // namespace tensorflow

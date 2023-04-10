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

#ifndef HYBRIDBACKEND_TENSORFLOW_DATA_TABULAR_TABLE_H_
#define HYBRIDBACKEND_TENSORFLOW_DATA_TABULAR_TABLE_H_

#include <memory>
#include <type_traits>
#include <vector>

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/types.h>

namespace tensorflow {
namespace hybridbackend {

enum TableFormat {
  kParquetFormat = 11,
  kOrcFormat = 21,
};

class TableAccess {
 public:
  static TableAccess* Create(
      OpKernelContext* ctx, const TableFormat& format, const string& filename,
      const int64 batch_size, const std::vector<string>& field_names,
      const DataTypeVector& field_dtypes,
      const std::vector<int32>& field_ragged_ranks,
      const std::vector<PartialTensorShape>& field_shapes,
      const bool drop_remainder, const bool skip_corrupted_data);

  TableAccess(const TableFormat& format, const string& filename,
              const int64 batch_size, const std::vector<string>& field_names,
              const DataTypeVector& field_dtypes,
              const std::vector<int32>& field_ragged_ranks,
              const std::vector<PartialTensorShape>& field_shapes,
              const bool drop_remainder, const bool skip_corrupted_data)
      : format_(format),
        filename_(std::move(filename)),
        batch_size_(batch_size),
        field_names_(std::move(field_names)),
        field_dtypes_(std::move(field_dtypes)),
        field_ragged_ranks_(std::move(field_ragged_ranks)),
        field_shapes_(std::move(field_shapes)),
        drop_remainder_(drop_remainder),
        skip_corrupted_data_(skip_corrupted_data) {}

  TableFormat format() const { return format_; }

  string filename() const { return filename_; }

  int64 batch_size() const { return batch_size_; }

  const std::vector<string>& field_names() const { return field_names_; }

  const DataTypeVector& field_dtypes() const { return field_dtypes_; }

  const std::vector<int32>& field_ragged_ranks() const {
    return field_ragged_ranks_;
  }

  const std::vector<PartialTensorShape>& field_shapes() const {
    return field_shapes_;
  }

  bool drop_remainder() const { return drop_remainder_; }

  bool skip_corrupted_data() const { return skip_corrupted_data_; }

  virtual int64 Count() const = 0;

  virtual Status Open() = 0;

  virtual Status Open(const int64 start, const int64 end) = 0;

  virtual Status Read(std::vector<Tensor>* output_tensors) = 0;

  virtual ~TableAccess() {}

 private:
  const TableFormat format_;
  const string filename_;
  const int64 batch_size_;
  const std::vector<string> field_names_;
  const DataTypeVector field_dtypes_;
  const std::vector<int32> field_ragged_ranks_;
  const std::vector<PartialTensorShape> field_shapes_;
  const bool drop_remainder_;
  const bool skip_corrupted_data_;
};

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW_DATA_TABULAR_TABLE_H_

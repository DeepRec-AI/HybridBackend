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

#ifndef HYBRIDBACKEND_TENSORFLOW_DATA_TABULAR_PARQUET_H_
#define HYBRIDBACKEND_TENSORFLOW_DATA_TABULAR_PARQUET_H_

#include <memory>
#include <vector>

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/types.h>

#include "hybridbackend/tensorflow/data/tabular/table.h"

namespace tensorflow {
namespace hybridbackend {

class ParquetAccess : public TableAccess {
 public:
  ParquetAccess(OpKernelContext* ctx, const TableFormat& format,
                const string& filename, const int64 batch_size,
                const std::vector<string>& field_names,
                const DataTypeVector& field_dtypes,
                const std::vector<int32>& field_ragged_ranks,
                const std::vector<PartialTensorShape>& field_shapes,
                const bool drop_remainder, const bool skip_corrupted_data);

  virtual int64 Count() const override;

  virtual Status Open() override;

  virtual Status Open(const int64 start, const int64 end) override;

  virtual Status Read(std::vector<Tensor>* output_tensors) override;

  virtual ~ParquetAccess();

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW_DATA_TABULAR_PARQUET_H_

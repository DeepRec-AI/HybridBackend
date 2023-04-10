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

#include <absl/strings/str_cat.h>
#include <memory>
#include <type_traits>
#include <vector>

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/types.h>

#include <unordered_set>

#include "hybridbackend/tensorflow/data/tabular/orc.h"
#include "hybridbackend/tensorflow/data/tabular/parquet.h"

namespace tensorflow {
namespace hybridbackend {

TableAccess* TableAccess::Create(
    OpKernelContext* ctx, const TableFormat& format, const string& filename,
    const int64 batch_size, const std::vector<string>& field_names,
    const DataTypeVector& field_dtypes,
    const std::vector<int32>& field_ragged_ranks,
    const std::vector<PartialTensorShape>& field_shapes,
    const bool drop_remainder, const bool skip_corrupted_data) {
  switch (format) {
    case kParquetFormat:
      return new ParquetAccess(ctx, format, filename, batch_size, field_names,
                               field_dtypes, field_ragged_ranks, field_shapes,
                               drop_remainder, skip_corrupted_data);
      break;
    case kOrcFormat:
      return new OrcAccess(ctx, format, filename, batch_size, field_names,
                           field_dtypes, field_ragged_ranks, field_shapes,
                           drop_remainder, skip_corrupted_data);
      break;
    default:
      LOG(ERROR) << "File format " << format << " is not supported";
      return nullptr;
  }
  return nullptr;
}

}  // namespace hybridbackend
}  // namespace tensorflow

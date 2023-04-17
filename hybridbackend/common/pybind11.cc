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

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if HYBRIDBACKEND_CUDA
#include <cuda.h>
#endif
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "hybridbackend/common/arrow.h"

namespace {
std::string make_buildinfo() {
  std::string message = "HybridBackend";
#if HYBRIDBACKEND_BUILDINFO
  message += " " HYBRIDBACKEND_BUILD_VERSION "-" HYBRIDBACKEND_BUILD_COMMIT "";
  message += "; " HYBRIDBACKEND_BUILD_FRAMEWORK "";
  message += "; " HYBRIDBACKEND_BUILD_CXX "";
  message += "; " HYBRIDBACKEND_BUILD_LOG "";
#if HYBRIDBACKEND_BUILD_CXX11_ABI > 0
  message += " (C++11 ABI)";
#endif
#if HYBRIDBACKEND_CUDA
  message += "; CUDA " + std::to_string(CUDA_VERSION / 1000) + "." +
             std::to_string((CUDA_VERSION / 10) % 100);
  message += " (" HYBRIDBACKEND_CUDA_GENCODE ")";
#endif
#endif
  return message;
}

std::string buildinfo() {
  static std::string kBuildInfo = make_buildinfo();
  return kBuildInfo;
}

typedef std::tuple<std::string, std::string, int> parquet_file_field_t;
std::vector<parquet_file_field_t> parquet_file_get_fields(
    const std::string& filename) {
#if HYBRIDBACKEND_ARROW
  std::vector<std::string> field_names;
  std::vector<std::string> field_dtypes;
  std::vector<int> field_ragged_ranks;
  auto s = ::hybridbackend::GetParquetDataFrameFields(
      &field_names, &field_dtypes, &field_ragged_ranks, filename);
  std::vector<parquet_file_field_t> fields;
  if (!s.ok()) {
    std::cerr << "parquet_file_get_fields failed: " << s.message() << std::endl;
    return fields;
  }
  for (size_t i = 0; i < field_names.size(); ++i) {
    fields.emplace_back(field_names[i], field_dtypes[i], field_ragged_ranks[i]);
  }
  return fields;
#else
  return {};
#endif
}

int parquet_file_count_row_groups(const std::string& filename) {
#if HYBRIDBACKEND_ARROW
  int row_group_count = -1;
  auto s = ::hybridbackend::GetParquetRowGroupCount(&row_group_count, filename);
  if (!s.ok()) {
    std::cerr << "parquet_file_count_row_groups failed: " << s.message()
              << std::endl;
    return -1;
  }
  return row_group_count;
#else
  return -1;
#endif
}

typedef std::tuple<std::string, std::string, int> orc_file_field_t;
std::vector<orc_file_field_t> orc_file_get_fields(const std::string& filename) {
#if HYBRIDBACKEND_ARROW
  std::vector<std::string> field_names;
  std::vector<std::string> field_dtypes;
  std::vector<int> field_ragged_ranks;
  auto s = ::hybridbackend::GetOrcDataFrameFields(
      &field_names, &field_dtypes, &field_ragged_ranks, filename);
  std::vector<orc_file_field_t> fields;
  if (!s.ok()) {
    std::cerr << "orc_file_get_fields failed: " << s.message() << std::endl;
    return fields;
  }
  for (size_t i = 0; i < field_names.size(); ++i) {
    fields.emplace_back(field_names[i], field_dtypes[i], field_ragged_ranks[i]);
  }
  return fields;
#else
  return {};
#endif
}

int orc_file_count_rows(const std::string& filename) {
#if HYBRIDBACKEND_ARROW
  int row_count = -1;
  auto s = ::hybridbackend::GetOrcRowCount(&row_count, filename);
  if (!s.ok()) {
    std::cerr << "orc_file_count_rows failed: " << s.message() << std::endl;
    return -1;
  }
  return row_count;
#else
  return -1;
#endif
}

}  // namespace

PYBIND11_MODULE(libhybridbackend, m) {
  m.def("buildinfo", &buildinfo, "Get building information.");
  m.def("parquet_file_get_fields", &parquet_file_get_fields,
        "Get fields of a Parquet file.");
  m.def("parquet_file_count_row_groups", &parquet_file_count_row_groups,
        "Get row group count of a Parquet file.");
  m.def("orc_file_get_fields", &orc_file_get_fields,
        "Get fields of a ORC file.");
  m.def("orc_file_count_rows", &orc_file_count_rows,
        "Get row count of a ORC file.");
}

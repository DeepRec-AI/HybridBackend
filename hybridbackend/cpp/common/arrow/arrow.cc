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

#include "hybridbackend/cpp/common/arrow/arrow.h"

#include <unistd.h>

#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if HYBRIDBACKEND_ARROW
#include <arrow/array.h>
#include <arrow/util/thread_pool.h>

#include "hybridbackend/cpp/common/env.h"

namespace hybridbackend {

#if HYBRIDBACKEND_ARROW

namespace {

int SetArrowCpuThreadPoolCapacityFromEnv() {
  int arrow_threads = EnvGetInt("ARROW_NUM_THREADS", 0);
  if (arrow_threads > 0) {  // Set from environment variable
    ::arrow::SetCpuThreadPoolCapacity(arrow_threads);
  }
  return arrow_threads;
}

::arrow::Status MakeNumpyDtypeAndRaggedRankFromArrowDataType(
    std::string* numpy_dtype, int* ragged_rank,
    const std::shared_ptr<::arrow::DataType>& arrow_dtype) {
  if (arrow_dtype->id() == ::arrow::Type::LIST) {
    ++(*ragged_rank);
    return MakeNumpyDtypeAndRaggedRankFromArrowDataType(
        numpy_dtype, ragged_rank, arrow_dtype->field(0)->type());
  }

  switch (arrow_dtype->id()) {
    case ::arrow::Type::INT8:
    case ::arrow::Type::UINT8:
    case ::arrow::Type::INT32:
    case ::arrow::Type::INT64:
    case ::arrow::Type::UINT64:
      *numpy_dtype = arrow_dtype->name();
      break;
    case ::arrow::Type::HALF_FLOAT:
      *numpy_dtype = "float16";
      break;
    case ::arrow::Type::FLOAT:
      *numpy_dtype = "float32";
      break;
    case ::arrow::Type::DOUBLE:
      *numpy_dtype = "float64";
      break;
    case ::arrow::Type::STRING:
      *numpy_dtype = "O";
      break;
    default:
      return ::arrow::Status::Invalid(
          "Arrow data type ", arrow_dtype->ToString(), " not supported.");
  }
  return ::arrow::Status::OK();
}

}  // namespace

int UpdateArrowCpuThreadPoolCapacityFromEnv() {
  static int arrow_threads = SetArrowCpuThreadPoolCapacityFromEnv();
  return arrow_threads;
}

int GetArrowFileBufferSizeFromEnv() {
  static int buffer_size = EnvGetInt("ARROW_FILE_BUFFER_SIZE", 4096 * 4);
  return buffer_size;
}

::arrow::Status OpenArrowFile(
    std::shared_ptr<::arrow::io::RandomAccessFile>* file,
    const std::string& filename) {
#if HYBRIDBACKEND_ARROW_HDFS
  if (filename.rfind("hdfs://", 0) == 0) {
    ::arrow::internal::Uri uri;
    ARROW_RETURN_NOT_OK(uri.Parse(filename));
    ARROW_ASSIGN_OR_RAISE(auto options, ::arrow::fs::HdfsOptions::FromUri(uri));
    std::shared_ptr<::arrow::io::HadoopFileSystem> fs;
    ARROW_RETURN_NOT_OK(::arrow::io::HadoopFileSystem::Connect(
        &options.connection_config, &fs));
    std::shared_ptr<::arrow::io::HdfsReadableFile> hdfs_file;
    ARROW_RETURN_NOT_OK(fs->OpenReadable(uri.path(), &hdfs_file));
    *file = hdfs_file;
    return ::arrow::Status::OK();
  }
#endif
#if HYBRIDBACKEND_ARROW_S3
  if (filename.rfind("s3://", 0) == 0 || filename.rfind("oss://", 0) == 0) {
    ARROW_RETURN_NOT_OK(::arrow::fs::EnsureS3Initialized());
    ::arrow::internal::Uri uri;
    ARROW_RETURN_NOT_OK(uri.Parse(filename));
    std::string path;
    ARROW_ASSIGN_OR_RAISE(auto options,
                          ::arrow::fs::S3Options::FromUri(uri, &path));
    ARROW_ASSIGN_OR_RAISE(auto fs, ::arrow::fs::S3FileSystem::Make(options));
    ARROW_ASSIGN_OR_RAISE(*file, fs->OpenInputFile(path));
    return ::arrow::Status::OK();
  }
#endif
  auto fs = std::make_shared<::arrow::fs::LocalFileSystem>();
  ARROW_ASSIGN_OR_RAISE(*file, fs->OpenInputFile(filename));
  return ::arrow::Status::OK();
}

::arrow::Status OpenParquetReader(
    std::unique_ptr<::parquet::arrow::FileReader>* reader,
    const std::shared_ptr<::arrow::io::RandomAccessFile>& file) {
  auto config = ::parquet::ReaderProperties();
  config.enable_buffered_stream();
  config.set_buffer_size(GetArrowFileBufferSizeFromEnv());
  ARROW_RETURN_NOT_OK(::parquet::arrow::FileReader::Make(
      ::arrow::default_memory_pool(),
      ::parquet::ParquetFileReader::Open(file, config), reader));
  // If ARROW_NUM_THREADS > 0, specified number of threads will be used.
  // If ARROW_NUM_THREADS = 0, no threads will be used.
  // If ARROW_NUM_THREADS < 0, all threads will be used.
  (*reader)->set_use_threads(UpdateArrowCpuThreadPoolCapacityFromEnv() != 0);
  return ::arrow::Status::OK();
}

::arrow::Status GetParquetDataFrameFields(
    std::vector<std::string>* field_names,
    std::vector<std::string>* field_dtypes,
    std::vector<int>* field_ragged_ranks, const std::string& filename) {
  std::shared_ptr<::arrow::io::RandomAccessFile> file;
  ARROW_RETURN_NOT_OK(::hybridbackend::OpenArrowFile(&file, filename));
  std::unique_ptr<::parquet::arrow::FileReader> reader;
  ARROW_RETURN_NOT_OK(::hybridbackend::OpenParquetReader(&reader, file));

  std::shared_ptr<::arrow::Schema> schema;
  ARROW_RETURN_NOT_OK(reader->GetSchema(&schema));
  if (ARROW_PREDICT_FALSE(!schema->HasDistinctFieldNames())) {
    return ::arrow::Status::Invalid(filename,
                                    " must has distinct column names");
  }
  for (const auto& field : schema->fields()) {
    field_names->push_back(field->name());
    std::string dtype;
    int ragged_rank = 0;
    ARROW_RETURN_NOT_OK(MakeNumpyDtypeAndRaggedRankFromArrowDataType(
        &dtype, &ragged_rank, field->type()));
    field_dtypes->push_back(dtype);
    field_ragged_ranks->push_back(ragged_rank);
  }
  return ::arrow::Status::OK();
}

#endif

}  // namespace hybridbackend

#endif  // HYBRIDBACKEND_ARROW

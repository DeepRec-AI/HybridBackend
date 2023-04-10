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

#ifndef HYBRIDBACKEND_COMMON_ARROW_H_
#define HYBRIDBACKEND_COMMON_ARROW_H_

#include <deque>
#include <string>

#if HYBRIDBACKEND_ARROW
#include <arrow/adapters/orc/adapter.h>
#include <arrow/dataset/api.h>
#include <arrow/record_batch.h>
#include <parquet/arrow/reader.h>
#include <parquet/properties.h>

#if HYBRIDBACKEND_ARROW_HDFS
#include <arrow/filesystem/hdfs.h>
#endif
#if HYBRIDBACKEND_ARROW_S3
#include <arrow/filesystem/s3fs.h>
#endif
#include <arrow/filesystem/localfs.h>

namespace hybridbackend {
::arrow::Status OpenArrowFile(
    std::shared_ptr<::arrow::fs::FileSystem>* fs,
    std::shared_ptr<::arrow::io::RandomAccessFile>* file,
    const std::string& filename);

void CloseArrowFile(std::shared_ptr<::arrow::fs::FileSystem>& fs,
                    std::shared_ptr<::arrow::io::RandomAccessFile>& file,
                    const std::string& filename);

::arrow::Status OpenParquetReader(
    std::unique_ptr<::parquet::arrow::FileReader>* reader,
    const std::shared_ptr<::arrow::io::RandomAccessFile>& file,
    const bool initialized_from_env);

::arrow::Status GetParquetDataFrameFields(
    std::vector<std::string>* field_names,
    std::vector<std::string>* field_dtypes,
    std::vector<int>* field_ragged_ranks, const std::string& filename);

::arrow::Status GetParquetRowGroupCount(int* row_group_count,
                                        const std::string& filename);

::arrow::Status OpenOrcReader(
    std::unique_ptr<::arrow::adapters::orc::ORCFileReader>* reader,
    const std::shared_ptr<::arrow::io::RandomAccessFile>& file,
    const bool initialized_from_env);

::arrow::Status GetOrcDataFrameFields(std::vector<std::string>* field_names,
                                      std::vector<std::string>* field_dtypes,
                                      std::vector<int>* field_ragged_ranks,
                                      const std::string& filename);

::arrow::Status GetOrcRowCount(int* row_count, const std::string& filename);

}  // namespace hybridbackend

#endif  // HYBRIDBACKEND_ARROW
#endif  // HYBRIDBACKEND_COMMON_ARROW_H_

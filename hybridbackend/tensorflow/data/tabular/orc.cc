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

#include "hybridbackend/tensorflow/data/tabular/orc.h"

#include <absl/strings/match.h>

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "hybridbackend/common/arrow.h"
#include "hybridbackend/tensorflow/common/arrow.h"

namespace tensorflow {
namespace hybridbackend {
class OrcAccess::Impl {
 public:
  Impl(OpKernelContext* ctx, const string& filename, const int64 batch_size,
       const std::vector<string>& field_names,
       const DataTypeVector& field_dtypes,
       const std::vector<int32>& field_ragged_ranks)
      : field_names_(field_names),
        batch_size_(batch_size),
        row_(0),
        end_(0),
        start_row_(-1),
        end_row_(-1) {
    OP_REQUIRES_OK(ctx, Initialize(filename, field_names, field_dtypes,
                                   field_ragged_ranks));
  }

  Status Initialize(const string& filename,
                    const std::vector<string>& field_names,
                    const DataTypeVector& field_dtypes,
                    const std::vector<int32>& field_ragged_ranks) {
#if HYBRIDBACKEND_ARROW
    ::arrow::internal::Uri uri;
    if (::arrow::fs::internal::IsLikelyUri(filename)) {
      TF_RETURN_IF_ARROW_ERROR(uri.Parse(filename));
    } else {
      if (TF_PREDICT_FALSE(
              !::arrow::fs::internal::IsLikelyUri("file://" + filename))) {
        return errors::InvalidArgument("File name ", filename, " is illegal");
      }
      TF_RETURN_IF_ARROW_ERROR(uri.Parse("file://" + filename));
    }
    std::vector<std::pair<std::string, std::string>> uri_options;
    TF_RETURN_IF_ARROW_ERROR(uri.query_items().Value(&uri_options));
    for (auto q : uri_options) {
      if (q.first == "start") {
        start_row_ = atoi(q.second.c_str());
      } else if (q.first == "end") {
        end_row_ = atoi(q.second.c_str());
      }
    }

    TF_RETURN_IF_ARROW_ERROR(
        ::hybridbackend::OpenArrowFile(&fs_, &file_, filename));
    TF_RETURN_IF_ARROW_ERROR(
        ::hybridbackend::OpenOrcReader(&reader_, file_, true));

    std::shared_ptr<::arrow::Schema> schema;
    TF_RETURN_IF_ARROW_ERROR(reader_->ReadSchema().Value(&schema));
    TF_RETURN_IF_ERROR(ValidateSchema(filename, field_names, field_dtypes,
                                      field_ragged_ranks, schema, &columns_));
#endif
    return Status::OK();
  }

  int64 Count() const {
#if HYBRIDBACKEND_ARROW
    if (TF_PREDICT_FALSE(start_row_ > -1 && end_row_ > -1)) {
      return end_row_ - start_row_;
    }
    return reader_->NumberOfRows();
#else
    return Status::OK();
#endif
  }

  Status Open() {
#if HYBRIDBACKEND_ARROW
    if (TF_PREDICT_TRUE(batch_reader_)) {
      return Status::OK();
    }

    if (TF_PREDICT_FALSE(start_row_ > -1 && end_row_ > -1)) {
      return Open(0, end_row_ - start_row_);
    }
    TF_RETURN_IF_ARROW_ERROR(
        reader_->GetRecordBatchReader(batch_size_, field_names_)
            .Value(&batch_reader_));
#endif
    return Status::OK();
  }

  Status Open(const int64 start, const int64 end) {
#if HYBRIDBACKEND_ARROW
    if (TF_PREDICT_TRUE(batch_reader_)) {
      return Status::OK();
    }
    row_ = start;
    end_ = end;
    reader_->Seek(start_row_ > -1 ? start_row_ + start : start);
    TF_RETURN_IF_ARROW_ERROR(
        reader_->NextStripeReader(batch_size_, columns_).Value(&batch_reader_));
#endif
    return Status::OK();
  }

  void Close(const string& filename) {
#if HYBRIDBACKEND_ARROW
    batch_reader_.reset();
    reader_.reset();
    ::hybridbackend::CloseArrowFile(fs_, file_, filename);
#endif
  }

  Status Read(const string& filename, const int64 batch_size,
              const std::vector<string>& field_names,
              const DataTypeVector& field_dtypes,
              const std::vector<int32>& field_ragged_ranks,
              const std::vector<PartialTensorShape>& field_shapes,
              const bool drop_remainder, std::vector<Tensor>* output_tensors) {
#if HYBRIDBACKEND_ARROW
    auto s =
        ReadRecordBatch(batch_reader_.get(), filename, batch_size, field_names,
                        field_dtypes, field_ragged_ranks, field_shapes,
                        drop_remainder, end_, output_tensors, &row_counter_);
    while (TF_PREDICT_FALSE(errors::IsOutOfRange(s)) && row_counter_ < end_) {
      TF_RETURN_IF_ARROW_ERROR(reader_->NextStripeReader(batch_size_, columns_)
                                   .Value(&batch_reader_));
      if (!batch_reader_) {
        Close(filename);
        return s;
      }
      s = ReadRecordBatch(batch_reader_.get(), filename, batch_size,
                          field_names, field_dtypes, field_ragged_ranks,
                          field_shapes, drop_remainder, end_, output_tensors,
                          &row_counter_);
    }
    return s;
#else
    return Status::OK();
#endif
  }

 private:
#if HYBRIDBACKEND_ARROW
  int64 row_counter_;
  std::shared_ptr<::arrow::fs::FileSystem> fs_;
  std::shared_ptr<::arrow::io::RandomAccessFile> file_;
  std::unique_ptr<arrow::adapters::orc::ORCFileReader> reader_;
  std::shared_ptr<::arrow::RecordBatchReader> batch_reader_;
  std::vector<string> field_names_;
  std::vector<int> columns_;
  int64 batch_size_;
  int64 row_;
  int64 end_;
  int64 start_row_;
  int64 end_row_;
#endif
};

OrcAccess::OrcAccess(OpKernelContext* ctx, const TableFormat& format,
                     const string& filename, const int64 batch_size,
                     const std::vector<string>& field_names,
                     const DataTypeVector& field_dtypes,
                     const std::vector<int32>& field_ragged_ranks,
                     const std::vector<PartialTensorShape>& field_shapes,
                     const bool drop_remainder, const bool skip_corrupted_data)
    : TableAccess(format, filename, batch_size, field_names, field_dtypes,
                  field_ragged_ranks, field_shapes, drop_remainder,
                  skip_corrupted_data),
      pimpl_(new OrcAccess::Impl(ctx, filename, batch_size, field_names,
                                 field_dtypes, field_ragged_ranks)) {}

int64 OrcAccess::Count() const { return pimpl_->Count(); }

Status OrcAccess::Open() { return pimpl_->Open(); }

Status OrcAccess::Open(const int64 start, const int64 end) {
  return pimpl_->Open(start, end);
}

Status OrcAccess::Read(std::vector<Tensor>* output_tensors) {
  return pimpl_->Read(filename(), batch_size(), field_names(), field_dtypes(),
                      field_ragged_ranks(), field_shapes(), drop_remainder(),
                      output_tensors);
}

OrcAccess::~OrcAccess() { pimpl_->Close(filename()); }

}  // namespace hybridbackend
}  // namespace tensorflow

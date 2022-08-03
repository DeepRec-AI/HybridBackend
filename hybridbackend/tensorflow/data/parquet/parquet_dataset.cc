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

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/framework/partial_tensor_shape.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/platform/file_system.h>

#include <unordered_set>

#include "hybridbackend/tensorflow/common/dataset.h"
#include "hybridbackend/tensorflow/data/parquet/batch_reader.h"

namespace tensorflow {
namespace hybridbackend {

REGISTER_OP("HbParquetTabularDataset")
    .Output("handle: variant")
    .Input("filename: string")
    .Input("batch_size: int64")
    .Attr("field_names: list(string) >= 1")
    .Attr("field_dtypes: list(type) >= 1")
    .Attr("field_ragged_ranks: list(int) >= 1")
    .Attr("field_shapes: list(shape) >= 1")
    .Attr("partition_count: int = 1")
    .Attr("partition_index: int = 0")
    .Attr("drop_remainder: bool = false")
    .SetIsStateful()  // NOTE: Source dataset ops must be marked stateful to
                      // inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
A dataset that outputs batches from a parquet file.

handle: The handle to reference the dataset.
filename: Path of file to read.
batch_size: Maxium number of samples in an output batch.
field_names: List of field names to read.
field_dtypes: List of data types for each field.
field_ragged_ranks: List of ragged rank for each field.
field_shapes: List of shapes for each field.
partition_count: Count of row group partitions.
partition_index: Index of row group partitions.
drop_remainder: If True, only keep batches with exactly `batch_size` samples.
)doc");

class ParquetTabularDatasetOp : public DatasetOpKernel {
 public:
  explicit ParquetTabularDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx),
        partition_count_(1),
        partition_index_(0),
        drop_remainder_(false) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_names", &field_names_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_dtypes", &field_dtypes_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("field_ragged_ranks", &field_ragged_ranks_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_shapes", &field_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_count", &partition_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_index", &partition_index_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("drop_remainder", &drop_remainder_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;

  std::vector<string> field_names_;
  DataTypeVector field_dtypes_;
  std::vector<int32> field_ragged_ranks_;
  std::vector<PartialTensorShape> field_shapes_;
  int64 partition_count_;
  int64 partition_index_;
  bool drop_remainder_;
};

class ParquetTabularDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const string& filename, const int64 batch_size,
          const std::vector<string>& field_names,
          const DataTypeVector& field_dtypes,
          const std::vector<int32>& field_ragged_ranks,
          const std::vector<PartialTensorShape>& field_shapes,
          const int64 partition_count, const int64 partition_index,
          const bool drop_remainder)
      : DatasetBase(DatasetContext(ctx)),
        filename_(std::move(filename)),
        batch_size_(batch_size),
        field_names_(std::move(field_names)),
        field_dtypes_(std::move(field_dtypes)),
        field_ragged_ranks_(std::move(field_ragged_ranks)),
        field_shapes_(std::move(field_shapes)),
        partition_count_(partition_count),
        partition_index_(partition_index),
        drop_remainder_(drop_remainder) {
    const int64 actual_batch_size(drop_remainder ? batch_size : -1);
    int64 num_outputs = field_names.size();
    for (int64 i = 0; i < field_names.size(); ++i) {
      output_dtypes_.push_back(std::move(field_dtypes[i]));
      output_shapes_.push_back(PartialTensorShape({actual_batch_size})
                                   .Concatenate(field_shapes_[i]));
      for (int64 j = 0; j < field_ragged_ranks_[i]; ++j) {
        output_dtypes_.push_back(DT_INT32);
        output_shapes_.push_back(PartialTensorShape({actual_batch_size}));
      }
      num_outputs += field_ragged_ranks_[i];
    }

    reader_ = absl::make_unique<ParquetBatchReader>(
        filename_, batch_size_, field_names_, field_dtypes_,
        field_ragged_ranks_, field_shapes_, partition_count_, partition_index_,
        drop_remainder_);
  }

  Status Open() {
    VLOG(1) << "Starting to read " << filename_ << " ...";
    return reader_->Open();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;

  const DataTypeVector& output_dtypes() const override {
    return output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return "ParquetTabularDatasetOp::Dataset";
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* filename = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(filename_, &filename));
    Node* batch_size;
    TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
    AttrValue field_names;
    b->BuildAttrValue(field_names_, &field_names);
    AttrValue field_dtypes;
    b->BuildAttrValue(field_dtypes_, &field_dtypes);
    AttrValue field_ragged_ranks;
    b->BuildAttrValue(field_ragged_ranks_, &field_ragged_ranks);
    AttrValue field_shapes;
    b->BuildAttrValue(field_ragged_ranks_, &field_shapes);
    AttrValue partition_count;
    b->BuildAttrValue(partition_count_, &partition_count);
    AttrValue partition_index;
    b->BuildAttrValue(partition_index_, &partition_index);
    AttrValue drop_remainder;
    b->BuildAttrValue(drop_remainder_, &drop_remainder);
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {{0, filename}, {1, batch_size}}, {},
                      {{"field_names", field_names},
                       {"field_dtypes", field_dtypes},
                       {"field_ragged_ranks", field_ragged_ranks},
                       {"field_shapes", field_shapes},
                       {"partition_count", partition_count},
                       {"partition_index", partition_index},
                       {"drop_remainder", drop_remainder}},
                      output));
    return Status::OK();
  }

 private:
  class Iterator;
  const string filename_;
  const int64 batch_size_;
  const std::vector<string> field_names_;
  const DataTypeVector field_dtypes_;
  const std::vector<int32> field_ragged_ranks_;
  const std::vector<PartialTensorShape> field_shapes_;
  const int64 partition_count_;
  const int64 partition_index_;
  const bool drop_remainder_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  std::unique_ptr<ParquetBatchReader> reader_;
};

void ParquetTabularDatasetOp::MakeDataset(OpKernelContext* ctx,
                                          DatasetBase** output) {
  string filename;
  OP_REQUIRES_OK(ctx, PARSE_SCALAR(ctx, "filename", &filename));

  int64 batch_size = 0;
  OP_REQUIRES_OK(ctx, PARSE_SCALAR(ctx, "batch_size", &batch_size));
  OP_REQUIRES(ctx, batch_size > 0,
              errors::InvalidArgument("batch_size must be greater than zero."));

  Dataset* ds =
      new Dataset(ctx, filename, batch_size, field_names_, field_dtypes_,
                  field_ragged_ranks_, field_shapes_, partition_count_,
                  partition_index_, drop_remainder_);
  OP_REQUIRES_OK(ctx, ds->Open());
  *output = ds;
}

class ParquetTabularDatasetOp::Dataset::Iterator
    : public DatasetIterator<ParquetTabularDatasetOp::Dataset> {
 public:
  explicit Iterator(const Params& params)
      : DatasetIterator<ParquetTabularDatasetOp::Dataset>(params) {}

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    mutex_lock l(mu_);
    Status s = dataset()->reader_->Read(out_tensors);

    if (s.ok()) {
      *end_of_sequence = false;
      return s;
    }
    if (!errors::IsOutOfRange(s)) {
      return s;
    }
    *end_of_sequence = true;
    return Status::OK();
  }

 protected:
  Status SaveInternal(IteratorStateWriter* writer) override {
    return errors::Unimplemented("SaveInternal is currently not supported");
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
    return errors::Unimplemented("RestoreInternal is currently not supported");
  }

 private:
  mutex mu_;
};

std::unique_ptr<IteratorBase>
ParquetTabularDatasetOp::Dataset::MakeIteratorInternal(
    const string& prefix) const {
  return std::unique_ptr<IteratorBase>(
      new Iterator({this, absl::StrCat(prefix, "::ParquetTabular")}));
}

REGISTER_KERNEL_BUILDER(Name("HbParquetTabularDataset").Device(DEVICE_CPU),
                        ParquetTabularDatasetOp);

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("HbParquetTabularDataset");

}  // namespace hybridbackend
}  // namespace tensorflow

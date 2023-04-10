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

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/types.h>

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/framework/partial_tensor_shape.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/platform/file_system.h>

#include <unordered_set>

#include "hybridbackend/tensorflow/common/dataset.h"
#include "hybridbackend/tensorflow/data/tabular/table.h"

namespace tensorflow {
namespace hybridbackend {

REGISTER_OP("HbTabularDataset")
    .Output("handle: variant")
    .Input("filename: string")
    .Input("batch_size: int64")
    .Attr("format: int")
    .Attr("field_names: list(string) >= 1")
    .Attr("field_dtypes: list(type) >= 1")
    .Attr("field_ragged_ranks: list(int) >= 1")
    .Attr("field_shapes: list(shape) >= 1")
    .Attr("drop_remainder: bool = false")
    .Attr("skip_corrupted_data: bool = false")
    .Attr("partition_count: int = 1")
    .Attr("partition_index: int = 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
A dataset that outputs batches from a file.

handle: The handle to reference the dataset.
filename: Path of file to read.
batch_size: Maxium number of samples in an output batch.
format: File format to use.
field_names: List of field names to read.
field_dtypes: List of data types for each field.
field_ragged_ranks: List of ragged rank for each field.
field_shapes: List of shapes for each field.
drop_remainder: If True, only keep batches with exactly `batch_size` samples.
skip_corrupted_data: If True, ignore batches with data loss errors.
partition_count: Count of row group partitions.
partition_index: Index of row group partitions.
)doc");

class TabularDatasetOp : public DatasetOpKernel {
 public:
  explicit TabularDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx),
        drop_remainder_(false),
        skip_corrupted_data_(false),
        partition_count_(1),
        partition_index_(0) {
    int format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("format", &format));
    format_ = static_cast<TableFormat>(format);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_names", &field_names_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_dtypes", &field_dtypes_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("field_ragged_ranks", &field_ragged_ranks_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_shapes", &field_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("drop_remainder", &drop_remainder_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("skip_corrupted_data", &skip_corrupted_data_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_count", &partition_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_index", &partition_index_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;
  TableFormat format_;
  std::vector<string> field_names_;
  DataTypeVector field_dtypes_;
  std::vector<int32> field_ragged_ranks_;
  std::vector<PartialTensorShape> field_shapes_;
  bool drop_remainder_;
  bool skip_corrupted_data_;
  int64 partition_count_;
  int64 partition_index_;
};

class TabularDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, TableAccess* access,
          const int64 partition_count, const int64 partition_index)
      : DatasetBase(DatasetContext(ctx)),
        access_(access),
        partition_count_(partition_count),
        partition_index_(partition_index) {
    const int64 actual_batch_size(
        access_->drop_remainder() ? access_->batch_size() : -1);
    for (int64 i = 0; i < access_->field_names().size(); ++i) {
      output_dtypes_.push_back(std::move(access_->field_dtypes()[i]));
      output_shapes_.push_back(PartialTensorShape({actual_batch_size})
                                   .Concatenate(access_->field_shapes()[i]));
      for (int64 j = 0; j < access_->field_ragged_ranks()[i]; ++j) {
        output_dtypes_.push_back(DT_INT32);
        output_shapes_.push_back(PartialTensorShape({actual_batch_size}));
      }
    }
  }

  const DataTypeVector& output_dtypes() const override {
    return output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return absl::StrCat("TabularDataset<", access_->format(), ">(",
                        access_->filename(), ")");
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* filename = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(access_->filename(), &filename));
    Node* batch_size;
    TF_RETURN_IF_ERROR(b->AddScalar(access_->batch_size(), &batch_size));
    AttrValue format;
    b->BuildAttrValue(access_->format(), &format);
    AttrValue field_names;
    b->BuildAttrValue(access_->field_names(), &field_names);
    AttrValue field_dtypes;
    b->BuildAttrValue(access_->field_dtypes(), &field_dtypes);
    AttrValue field_ragged_ranks;
    b->BuildAttrValue(access_->field_ragged_ranks(), &field_ragged_ranks);
    AttrValue field_shapes;
    b->BuildAttrValue(access_->field_ragged_ranks(), &field_shapes);
    AttrValue drop_remainder;
    b->BuildAttrValue(access_->drop_remainder(), &drop_remainder);
    AttrValue skip_corrupted_data;
    b->BuildAttrValue(access_->skip_corrupted_data(), &skip_corrupted_data);
    AttrValue partition_count;
    b->BuildAttrValue(partition_count_, &partition_count);
    AttrValue partition_index;
    b->BuildAttrValue(partition_index_, &partition_index);
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {{0, filename}, {1, batch_size}}, {},
                      {{"format", format},
                       {"field_names", field_names},
                       {"field_dtypes", field_dtypes},
                       {"field_ragged_ranks", field_ragged_ranks},
                       {"field_shapes", field_shapes},
                       {"drop_remainder", drop_remainder},
                       {"skip_corrupted_data", skip_corrupted_data},
                       {"partition_count", partition_count},
                       {"partition_index", partition_index}},
                      output));
    return Status::OK();
  }

 private:
  class Iterator;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  std::unique_ptr<TableAccess> access_;
  int64 partition_count_;
  int64 partition_index_;
};

void TabularDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  string filename;
  OP_REQUIRES_OK(ctx, PARSE_SCALAR(ctx, "filename", &filename));

  int64 batch_size = 0;
  OP_REQUIRES_OK(ctx, PARSE_SCALAR(ctx, "batch_size", &batch_size));
  OP_REQUIRES(ctx, batch_size > 0,
              errors::InvalidArgument("batch_size must be greater than zero."));
  OP_REQUIRES(ctx, partition_index_ < partition_count_,
              errors::InvalidArgument("Partition index ", partition_index_,
                                      " must be smaller than partition count ",
                                      partition_count_));
  OP_REQUIRES(ctx, partition_index_ >= 0,
              errors::InvalidArgument("Partition index ", partition_index_,
                                      "must be greater than 0"));
  TableAccess* access =
      TableAccess::Create(ctx, format_, filename, batch_size, field_names_,
                          field_dtypes_, field_ragged_ranks_, field_shapes_,
                          drop_remainder_, skip_corrupted_data_);
  int64 count = access->Count();
  if (TF_PREDICT_FALSE(partition_count_ > 1)) {
    int64 partition_size = count / partition_count_;
    int full_partition_count = 0;
    if (TF_PREDICT_FALSE(partition_size < 1)) {
      partition_size = 1;
      full_partition_count = count;
    } else {
      full_partition_count = count / partition_size;
    }
    int64 start = 0;
    if (partition_index_ < full_partition_count) {
      start = partition_size * partition_index_;
    } else {
      start = partition_size * full_partition_count;
    }
    int64 end = 0;
    if (partition_index_ + 1 < full_partition_count) {
      end = partition_size * (partition_index_ + 1);
      if (end > count) {
        end = count;
      }
    } else {
      end = partition_size * full_partition_count;
    }
    OP_REQUIRES_OK(ctx, access->Open(start, end));
  } else {
    OP_REQUIRES_OK(ctx, access->Open());
  }
  *output = new TabularDatasetOp::Dataset(ctx, access, partition_count_,
                                          partition_index_);
}

class TabularDatasetOp::Dataset::Iterator
    : public DatasetIterator<TabularDatasetOp::Dataset> {
 public:
  explicit Iterator(const Params& params)
      : DatasetIterator<TabularDatasetOp::Dataset>(params) {}

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    mutex_lock l(mu_);
    Status s = dataset()->access_->Read(out_tensors);
    for (; dataset()->access_->skip_corrupted_data() && errors::IsDataLoss(s);
         s = dataset()->access_->Read(out_tensors)) {
      LOG(ERROR) << "Skip corrupted data: " << s.error_message();
      out_tensors->clear();
    }
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

std::unique_ptr<IteratorBase> TabularDatasetOp::Dataset::MakeIteratorInternal(
    const string& prefix) const {
  VLOG(1) << "Starting to read " << access_->filename() << " ("
          << access_->format() << ") in batches (<= " << access_->batch_size()
          << " samples/batch) ...";
  return std::unique_ptr<IteratorBase>(new TabularDatasetOp::Dataset::Iterator(
      {this, absl::StrCat(prefix, "::", access_->format(), "TabularDataset")}));
}

REGISTER_KERNEL_BUILDER(Name("HbTabularDataset").Device(DEVICE_CPU),
                        TabularDatasetOp);

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("HbTabularDataset");

}  // namespace hybridbackend
}  // namespace tensorflow

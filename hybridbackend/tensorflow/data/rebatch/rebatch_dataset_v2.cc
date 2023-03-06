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
#include <tensorflow/core/lib/core/blocking_counter.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/platform/file_system.h>
#include <tensorflow/core/public/version.h>

#include <deque>
#include <map>
#include <unordered_set>
#include <vector>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/common/dataset.h"
#include "hybridbackend/tensorflow/common/eigen.h"
#include "hybridbackend/tensorflow/data/rebatch/buffer.h"

namespace tensorflow {
namespace hybridbackend {

constexpr char kRebatchDatasetFastPathEnv[] =
    "HB_DATA_REBATCH_FASTPATH_DISABLED";

REGISTER_OP("HbRebatchTabularDatasetV2")
    .Output("handle: variant")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("shuffle_buffer_size: int64")
    .Input("shuffle_seed: int64")
    .Input("shuffle_seed2: int64")
    .Attr("drop_remainder: bool")
    .Attr("reshuffle_each_iteration: bool = true")
    .Attr("field_ids: list(int) >= 1")
    .Attr("field_ragged_indices: list(int) >= 1")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // shuffle_buffer_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      // shuffle_seed should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      // shuffle_seed2 should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
A dataset that resizes batches from another tabular dataset.

handle: The handle to reference the dataset.
input_dataset: Input batch dataset.
batch_size: Maxium number of samples in an output batch.
shuffle_buffer_size: Buffer size to shuffle.
shuffle_seed: Seed for shuffling.
shuffle_seed2: Seed for shuffling.
drop_remainder: If True, only keep batches with exactly `batch_size` samples.
reshuffle_each_iteration: If true, the dataset should be pseudorandomly
  reshuffled each time it is iterated over.
field_ids: A list of tensor indices to indicate the type of a tensor is
  values (0), batch splits (1) or other splits (>1).
field_ragged_indices: A list of indices to indicate the type of a tensor is
  values (0), batch splits (1) or other splits (>1).
)doc");

class RebatchTabularDatasetV2Op : public UnaryDatasetOpKernel {
 public:
  explicit RebatchTabularDatasetV2Op(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        drop_remainder_(false),
        reshuffle_each_iteration_(false) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("drop_remainder", &drop_remainder_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reshuffle_each_iteration",
                                     &reshuffle_each_iteration_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_ids", &field_ids_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("field_ragged_indices", &field_ragged_indices_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;

  bool drop_remainder_;
  bool reshuffle_each_iteration_;
  std::vector<int> field_ids_;
  std::vector<int> field_ragged_indices_;
};

class RebatchTabularDatasetV2Op::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input_dataset,
          const int64 batch_size, const bool drop_remainder,
          const int64 shuffle_buffer_size, const int64 shuffle_seed,
          const int64 shuffle_seed2, const bool reshuffle_each_iteration,
          const std::vector<int>& field_ids,
          const std::vector<int>& field_ragged_indices)
      : DatasetBase(DatasetContext(ctx)),
        input_dataset_(input_dataset),
        batch_size_(batch_size),
        drop_remainder_(drop_remainder),
        shuffle_buffer_size_(shuffle_buffer_size),
        shuffle_seed_(shuffle_seed),
        shuffle_seed2_(shuffle_seed2),
        reshuffle_each_iteration_(reshuffle_each_iteration),
        field_ids_(field_ids),
        field_ragged_indices_(field_ragged_indices) {
    input_dataset_->Ref();

    const auto& input_shapes = input_dataset_->output_shapes();
    output_shapes_.reserve(input_shapes.size());
    for (const auto& input_shape : input_shapes) {
      auto partial_shape = input_shape;
      partial_shape.RemoveDim(0);
      if (drop_remainder_) {
        output_shapes_.emplace_back(
            PartialTensorShape({batch_size_}).Concatenate(partial_shape));
      } else {
        output_shapes_.emplace_back(
            PartialTensorShape({-1}).Concatenate(partial_shape));
      }
    }

    int64 prev_field_id = -1;
    int64 prev_ragged_index = -1;
    for (size_t i = 0; i < field_ids.size(); ++i) {
      const int64 field_id = field_ids[i];
      const int64 ragged_index = field_ragged_indices[i];
      auto output_dtype = output_dtypes()[i];
      if (TF_PREDICT_FALSE(ragged_index > 1 && output_dtype != DT_INT32)) {
        LOG(ERROR) << "Output tensor " << i << " must be DT_INT32 not "
                   << DataTypeString(output_dtype);
        return;
      }
      if (ragged_index <= prev_ragged_index) {
        if (TF_PREDICT_FALSE(field_id == prev_field_id)) {
          LOG(ERROR) << "Invalid `fields` serialization";
          return;
        }
        field_ranks_.push_back(prev_ragged_index);
      }
      prev_field_id = field_id;
      prev_ragged_index = ragged_index;
    }
    field_ranks_.push_back(prev_ragged_index);
  }

  ~Dataset() override { input_dataset_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;

  const DataTypeVector& output_dtypes() const override {
    return input_dataset_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return "RebatchTabularDatasetV2Op::Dataset";
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_dataset = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_dataset_, &input_dataset));
    Node* batch_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
    Node* shuffle_buffer_size = nullptr;
    TF_RETURN_IF_ERROR(
        b->AddScalar(shuffle_buffer_size_, &shuffle_buffer_size));
    Node* shuffle_seed = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(shuffle_seed_, &shuffle_seed));
    Node* shuffle_seed2 = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(shuffle_seed2_, &shuffle_seed2));
    AttrValue drop_remainder;
    b->BuildAttrValue(drop_remainder_, &drop_remainder);
    AttrValue reshuffle_each_iteration;
    b->BuildAttrValue(reshuffle_each_iteration_, &reshuffle_each_iteration);
    AttrValue field_ids;
    b->BuildAttrValue(field_ids_, &field_ids);
    AttrValue field_ragged_indices;
    b->BuildAttrValue(field_ragged_indices_, &field_ragged_indices);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this,
                      {{0, input_dataset},
                       {1, batch_size},
                       {2, shuffle_buffer_size},
                       {3, shuffle_seed},
                       {4, shuffle_seed2}},
                      {},
                      {{"drop_remainder", drop_remainder},
                       {"reshuffle_each_iteration", reshuffle_each_iteration},
                       {"field_ids", field_ids},
                       {"field_ragged_indices", field_ragged_indices}},
                      output));
    return Status::OK();
  }

 private:
  class Iterator;
  const DatasetBase* const input_dataset_;
  const int64 batch_size_;
  const bool drop_remainder_;
  const int64 shuffle_buffer_size_;
  const int64 shuffle_seed_;
  const int64 shuffle_seed2_;
  const bool reshuffle_each_iteration_;
  const std::vector<int> field_ids_;
  const std::vector<int> field_ragged_indices_;

  std::vector<PartialTensorShape> output_shapes_;
  std::vector<int32> field_ranks_;
};

void RebatchTabularDatasetV2Op::MakeDataset(OpKernelContext* ctx,
                                            DatasetBase* input,
                                            DatasetBase** output) {
  int64 batch_size = 0;
  OP_REQUIRES_OK(ctx, PARSE_SCALAR(ctx, "batch_size", &batch_size));
  OP_REQUIRES(ctx, batch_size > 0,
              errors::InvalidArgument("batch_size must be greater than zero."));

  int64 shuffle_buffer_size = 0;
  OP_REQUIRES_OK(
      ctx, PARSE_SCALAR(ctx, "shuffle_buffer_size", &shuffle_buffer_size));
  int64 shuffle_seed = 0;
  OP_REQUIRES_OK(ctx, PARSE_SCALAR(ctx, "shuffle_seed", &shuffle_seed));
  int64 shuffle_seed2 = 0;
  OP_REQUIRES_OK(ctx, PARSE_SCALAR(ctx, "shuffle_seed2", &shuffle_seed2));

  *output =
      new Dataset(ctx, input, batch_size, drop_remainder_, shuffle_buffer_size,
                  shuffle_seed, shuffle_seed2, reshuffle_each_iteration_,
                  field_ids_, field_ragged_indices_);
}

class RebatchTabularDatasetV2Op::Dataset::Iterator
    : public DatasetIterator<RebatchTabularDatasetV2Op::Dataset> {
 public:
  explicit Iterator(const Params& params)
      : DatasetIterator<RebatchTabularDatasetV2Op::Dataset>(params),
        buffer_(dataset()->output_dtypes(), dataset()->output_shapes(),
                dataset()->field_ranks_),
        shuffle_seed_(dataset()->shuffle_seed_),
        shuffle_seed2_(dataset()->shuffle_seed2_),
        parent_generator_(dataset()->shuffle_seed_, dataset()->shuffle_seed2_),
        generator_(&parent_generator_) {
    if (dataset()->reshuffle_each_iteration_) {
      // TODO: support restore
      shuffle_seed_ = generator_();
      shuffle_seed2_ = generator_();
    }
  }

  Status Initialize(IteratorContext* ctx) override {
    return dataset()->input_dataset_->MakeIterator(ctx, prefix(), &input_impl_);
  }

  Status GetNextInternal(IteratorContext* ctx,
                         std::vector<Tensor>* output_tensors,
                         bool* end_of_sequence) override {
    mutex_lock l(mu_);
    Allocator* alloc = ctx->allocator({});

    if (buffer_.size() >= dataset()->batch_size_ &&
        buffer_.size() >= dataset()->shuffle_buffer_size_) {
      if (dataset()->shuffle_buffer_size_ > 0) {
        TF_RETURN_IF_ERROR(buffer_.Shuffle(generator_, dataset()->batch_size_));
      }
      return buffer_.Take(alloc, output_tensors, dataset()->batch_size_);
    }

    *end_of_sequence = false;
    while (input_impl_ && !*end_of_sequence) {
      if (buffer_.size() >= dataset()->batch_size_ &&
          buffer_.size() >= dataset()->shuffle_buffer_size_) {
        if (dataset()->shuffle_buffer_size_ > 0) {
          TF_RETURN_IF_ERROR(
              buffer_.Shuffle(generator_, dataset()->batch_size_));
        }
        return buffer_.Take(alloc, output_tensors, dataset()->batch_size_);
      }

      std::vector<Tensor> input_tensors;
      TF_RETURN_IF_ERROR(
          input_impl_->GetNext(ctx, &input_tensors, end_of_sequence));
      if (!*end_of_sequence) {
        for (size_t i = 0; i < input_tensors.size(); ++i) {
          if (input_tensors[i].dims() == 0) {
            return errors::InvalidArgument(
                "Input element must have a non-scalar value in each "
                "component.");
          }
        }

        // Compute input batch size
        int64 input_batch_size = 0;
        if (TF_PREDICT_TRUE(input_tensors.size() > 0)) {
          if (dataset()->field_ranks_[0] > 0) {
            input_batch_size = input_tensors[1].dim_size(0) - 1;
          } else {
            input_batch_size = input_tensors[0].dim_size(0);
          }
        }

        // Fast path for same batch size
        static const bool kRebatchFaspathDisabled =
            ::hybridbackend::EnvVarGetBool(kRebatchDatasetFastPathEnv, false);
        if (TF_PREDICT_TRUE(!kRebatchFaspathDisabled)) {
          if (dataset()->shuffle_buffer_size_ < 1 && buffer_.size() == 0 &&
              dataset()->batch_size_ == input_batch_size) {
            *output_tensors = std::move(input_tensors);
            return Status::OK();
          }
        }

        if (dataset()->shuffle_buffer_size_ > 0) {
          // Insert input rows to buffer
          for (int64 row = 0; row < input_batch_size; ++row) {
            TF_RETURN_IF_ERROR(buffer_.PutSlice(input_tensors, row, row + 1));
          }
        } else {
          // Insert input batch to buffer
          TF_RETURN_IF_ERROR(
              buffer_.Put(std::move(input_tensors), input_batch_size));
        }
      }
    }
    input_impl_.reset();
    if (buffer_.size() > dataset()->batch_size_) {
      *end_of_sequence = false;
      if (dataset()->shuffle_buffer_size_ > 0) {
        TF_RETURN_IF_ERROR(buffer_.Shuffle(generator_, dataset()->batch_size_));
      }
      return buffer_.Take(alloc, output_tensors, dataset()->batch_size_);
    }
    if (buffer_.size() > 0 && !dataset()->drop_remainder_) {
      *end_of_sequence = false;
      if (dataset()->shuffle_buffer_size_ > 0) {
        TF_RETURN_IF_ERROR(buffer_.Shuffle(generator_, buffer_.size()));
      }
      return buffer_.Take(alloc, output_tensors, buffer_.size());
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
  std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
  RebatchBuffer buffer_ GUARDED_BY(mu_);

  int64 shuffle_seed_;
  int64 shuffle_seed2_;
  random::PhiloxRandom parent_generator_;
  random::SingleSampleAdapter<random::PhiloxRandom> generator_;
};

std::unique_ptr<IteratorBase>
RebatchTabularDatasetV2Op::Dataset::MakeIteratorInternal(
    const string& prefix) const {
  return std::unique_ptr<IteratorBase>(
      new Iterator({this, absl::StrCat(prefix, "::RebatchTabularV2")}));
}

REGISTER_KERNEL_BUILDER(Name("HbRebatchTabularDatasetV2").Device(DEVICE_CPU),
                        RebatchTabularDatasetV2Op);

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("HbRebatchTabularDatasetV2");

}  // namespace hybridbackend
}  // namespace tensorflow

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
#include <tensorflow/core/public/version.h>

#include <deque>
#include <vector>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/common/dataset.h"
#include "hybridbackend/tensorflow/data/rectify/queue.h"

namespace tensorflow {
namespace hybridbackend {

constexpr char kRectifyDatasetWorkerPool[] = "rectify_dataset_worker_pool";
constexpr char kRectifyDatasetFastPathEnv[] =
    "HB_DATA_RECTIFY_FAST_PATH_DISABLED";

REGISTER_OP("HbRectifyDataset")
    .Output("handle: variant")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("shuffle_buffer_size: int64")
    .Input("shuffle_seed: int64")
    .Input("shuffle_seed2: int64")
    .Attr("drop_remainder: bool")
    .Attr("reshuffle_each_iteration: bool = true")
    .Attr("output_kinds: list(int) >= 1")
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
A dataset that rectifies samples.

handle: The handle to reference the dataset.
input_dataset: Input batch dataset.
batch_size: Maxium number of samples in an output batch.
drop_remainder: If True, only keep batches with exactly `batch_size` samples.
)doc");

class RectifyDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit RectifyDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx), drop_remainder_(false) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("drop_remainder", &drop_remainder_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reshuffle_each_iteration",
                                     &reshuffle_each_iteration_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_kinds", &output_kinds_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;

  bool drop_remainder_;
  bool reshuffle_each_iteration_;
  std::vector<int> output_kinds_;
};

class RectifyDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input_dataset,
          const int64 batch_size, const bool drop_remainder,
          const int64 shuffle_buffer_size, const int64 shuffle_seed,
          const int64 shuffle_seed2, const bool reshuffle_each_iteration,
          const std::vector<int>& output_kinds)
      : DatasetBase(DatasetContext(ctx)),
        input_dataset_(input_dataset),
        batch_size_(batch_size),
        drop_remainder_(drop_remainder),
        shuffle_buffer_size_(shuffle_buffer_size),
        shuffle_seed_(shuffle_seed),
        shuffle_seed2_(shuffle_seed2),
        reshuffle_each_iteration_(reshuffle_each_iteration),
        output_kinds_(output_kinds) {
    input_dataset_->Ref();

    const auto& input_shapes = input_dataset_->output_shapes();
    output_shapes_.reserve(input_shapes.size());
    for (size_t i = 0; i < input_shapes.size(); ++i) {
      auto kind = output_kinds[i];
      if (kind == kTensorOrSparseTensorValues) {
        auto partial_shape = input_shapes[i];
        partial_shape.RemoveDim(0);
        if (drop_remainder_) {
          output_shapes_.emplace_back(
              PartialTensorShape({batch_size_}).Concatenate(partial_shape));
        } else {
          output_shapes_.emplace_back(
              PartialTensorShape({-1}).Concatenate(partial_shape));
        }
      } else {
        output_shapes_.emplace_back(input_shapes[i]);
      }
    }
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

  const std::vector<int>& output_kinds() const { return output_kinds_; }

  string DebugString() const override { return "RectifyDatasetOp::Dataset"; }

  int64 Cardinality() const override {
    int64 n = input_dataset_->Cardinality();
    if (n == data::kInfiniteCardinality || n == data::kUnknownCardinality) {
      return n;
    }
    return data::kUnknownCardinality;
  }

  Status CheckExternalState() const override {
    return input_dataset_->CheckExternalState();
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
    AttrValue output_kinds;
    b->BuildAttrValue(output_kinds_, &output_kinds);

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
                       {"output_kinds", output_kinds}},
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

  const std::vector<int> output_kinds_;
  std::vector<PartialTensorShape> output_shapes_;
};

void RectifyDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
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

  *output = new Dataset(ctx, input, batch_size, drop_remainder_,
                        shuffle_buffer_size, shuffle_seed, shuffle_seed2,
                        reshuffle_each_iteration_, output_kinds_);
}

class RectifyDatasetOp::Dataset::Iterator
    : public DatasetIterator<RectifyDatasetOp::Dataset> {
 public:
  explicit Iterator(const Params& params)
      : DatasetIterator<RectifyDatasetOp::Dataset>(params),
        queue_(dataset()->shuffle_buffer_size_, dataset()->shuffle_seed_,
               dataset()->shuffle_seed2_, dataset()->reshuffle_each_iteration_,
               dataset()->output_dtypes(), dataset()->output_shapes(),
               dataset()->output_kinds()),
        first_values_idx_(0) {
    for (first_values_idx_ = 0;
         first_values_idx_ < dataset()->output_kinds().size();
         ++first_values_idx_) {
      if (dataset()->output_kinds()[first_values_idx_] ==
          kTensorOrSparseTensorValues) {
        break;
      }
    }
    DCHECK(first_values_idx_ < dataset()->output_kinds().size());
  }

  Status Initialize(IteratorContext* ctx) override {
    return dataset()->input_dataset_->MakeIterator(ctx, prefix(), &input_impl_);
  }

  Status GetNextInternal(IteratorContext* ctx,
                         std::vector<Tensor>* output_tensors,
                         bool* end_of_sequence) override {
    mutex_lock l(mu_);
    Allocator* alloc = ctx->allocator({});

    int64 batch_size = dataset()->batch_size_;
    int64 shuffle_buffer_size =
        std::max(batch_size, dataset()->shuffle_buffer_size_);
    if (queue_.size() >= shuffle_buffer_size) {
      return queue_.Pop(batch_size, output_tensors, alloc);
    }

    if (!input_impl_) {
      if (queue_.size() > batch_size) {
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(queue_.Pop(batch_size, output_tensors, alloc));
        return Status::OK();
      }
      if (!dataset()->drop_remainder_ && queue_.size() > 0) {
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(queue_.Pop(queue_.size(), output_tensors, alloc));
        return Status::OK();
      }
      *end_of_sequence = true;
      return Status::OK();
    }

    *end_of_sequence = false;
    while (!*end_of_sequence) {
      if (queue_.size() >= shuffle_buffer_size) {
        return queue_.Pop(batch_size, output_tensors, alloc);
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
        const int64 input_batch_size =
            input_tensors[first_values_idx_].dim_size(0);
        // Fast path for same batch size
        static const bool kRectifyFastPathDisabled =
            ::hybridbackend::EnvVarGetBool(kRectifyDatasetFastPathEnv, false);
        if (TF_PREDICT_TRUE(!kRectifyFastPathDisabled)) {
          if (dataset()->shuffle_buffer_size_ < 1 && queue_.size() == 0 &&
              batch_size == input_batch_size) {
            *output_tensors = std::move(input_tensors);
            return Status::OK();
          }
        }
        queue_.Push(input_batch_size, std::move(input_tensors));
      }
    }
    if (queue_.size() > batch_size) {
      *end_of_sequence = false;
      input_impl_.reset();
      return queue_.Pop(batch_size, output_tensors, alloc);
    }
    if (!dataset()->drop_remainder_ && queue_.size() > 0) {
      *end_of_sequence = false;
      input_impl_.reset();
      return queue_.Pop(queue_.size(), output_tensors, alloc);
    }
    input_impl_.reset();
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
  RectifyQueue queue_ GUARDED_BY(mu_);
  int64 first_values_idx_ GUARDED_BY(mu_);
};

std::unique_ptr<IteratorBase> RectifyDatasetOp::Dataset::MakeIteratorInternal(
    const string& prefix) const {
  return std::unique_ptr<IteratorBase>(
      new Iterator({this, absl::StrCat(prefix, "::Rectify")}));
}

REGISTER_KERNEL_BUILDER(Name("HbRectifyDataset").Device(DEVICE_CPU),
                        RectifyDatasetOp);

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("HbRectifyDataset");

}  // namespace hybridbackend
}  // namespace tensorflow

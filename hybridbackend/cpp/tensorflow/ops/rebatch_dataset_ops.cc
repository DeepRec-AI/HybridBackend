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

#include <map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/public/version.h"

#include "hybridbackend/cpp/tensorflow/eigen.h"
#include "hybridbackend/cpp/tensorflow/io/dataset.h"

namespace tensorflow {
namespace hybridbackend {

constexpr char kRebatchTabularDatasetWorkerPool[] =
    "rebatch_tabular_dataset_worker_pool";

REGISTER_OP("RebatchTabularDataset")
    .Output("handle: variant")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("min_batch_size: int64")
    .Attr("field_ids: list(int) >= 1")
    .Attr("field_ragged_indices: list(int) >= 1")
    .Attr("drop_remainder: bool")
    .Attr("num_parallel_scans: int = 1")
    .SetIsStateful()  // NOTE: Source dataset ops must be marked stateful to
                      // inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // min_batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
A dataset that resizes batches from another tabular dataset.

handle: The handle to reference the dataset.
input_dataset: Input batch dataset.
batch_size: Maxium number of samples in an output batch.
min_batch_size: Minimum number of samples in an non-final batch.
field_ids: A list of tensor indices to indicate the type of a tensor is
  values (0), batch splits (1) or other splits (>1).
field_ragged_indices: A list of indices to indicate the type of a tensor is
  values (0), batch splits (1) or other splits (>1).
drop_remainder: If True, only keep batches with exactly `batch_size` samples.
num_parallel_scans: Number of concurrent scans against fields of input dataset.
)doc");

class RebatchTabularDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit RebatchTabularDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        drop_remainder_(false),
        num_parallel_scans_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_ids", &field_ids_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("field_ragged_indices", &field_ragged_indices_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("drop_remainder", &drop_remainder_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("num_parallel_scans", &num_parallel_scans_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;

  std::vector<int> field_ids_;
  std::vector<int> field_ragged_indices_;
  bool drop_remainder_;
  int num_parallel_scans_;
};

class RebatchTabularDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input_dataset,
          const int64 batch_size, const int64 min_batch_size,
          const std::vector<int>& field_ids,
          const std::vector<int>& field_ragged_indices,
          const bool drop_remainder, const int num_parallel_scans)
      : DatasetBase(DatasetContext(ctx)),
        input_dataset_(input_dataset),
        batch_size_(batch_size),
        min_batch_size_(min_batch_size),
        field_ids_(field_ids),
        field_ragged_indices_(field_ragged_indices),
        drop_remainder_(drop_remainder),
        num_parallel_scans_(num_parallel_scans) {
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
    return "RebatchTabularDatasetOp::Dataset";
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_dataset = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_dataset_, &input_dataset));
    Node* batch_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
    Node* min_batch_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(min_batch_size_, &min_batch_size));
    AttrValue field_ids;
    b->BuildAttrValue(field_ids_, &field_ids);
    AttrValue field_ragged_indices;
    b->BuildAttrValue(field_ragged_indices_, &field_ragged_indices);
    AttrValue drop_remainder;
    b->BuildAttrValue(drop_remainder_, &drop_remainder);
    AttrValue num_parallel_scans;
    b->BuildAttrValue(num_parallel_scans_, &num_parallel_scans);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {{0, input_dataset}, {1, batch_size}, {2, min_batch_size}}, {},
        {{"field_ids", field_ids},
         {"field_ragged_indices", field_ragged_indices},
         {"drop_remainder", drop_remainder},
         {"num_parallel_scans", num_parallel_scans}},
        output));
    return Status::OK();
  }

 private:
  class Iterator;
  const DatasetBase* const input_dataset_;
  const int64 batch_size_;
  const int64 min_batch_size_;
  const std::vector<int> field_ids_;
  const std::vector<int> field_ragged_indices_;
  const bool drop_remainder_;
  const int num_parallel_scans_;

  std::vector<PartialTensorShape> output_shapes_;
};

void RebatchTabularDatasetOp::MakeDataset(OpKernelContext* ctx,
                                          DatasetBase* input,
                                          DatasetBase** output) {
  int64 batch_size = 0;
  OP_REQUIRES_OK(ctx, PARSE_SCALAR(ctx, "batch_size", &batch_size));
  OP_REQUIRES(ctx, batch_size > 0,
              errors::InvalidArgument("batch_size must be greater than zero."));

  int64 min_batch_size = 0;
  OP_REQUIRES_OK(ctx, PARSE_SCALAR(ctx, "min_batch_size", &min_batch_size));
  OP_REQUIRES(
      ctx, min_batch_size > 0,
      errors::InvalidArgument("min_batch_size must be greater than zero."));

  OP_REQUIRES(ctx, batch_size >= min_batch_size,
              errors::InvalidArgument(
                  "batch_size must be greater than min_batch_size."));

  *output =
      new Dataset(ctx, input, batch_size, min_batch_size, field_ids_,
                  field_ragged_indices_, drop_remainder_, num_parallel_scans_);
}

namespace {
void RecalculateSplit(Tensor* split, int32 value) {
  int32* sdata =
      reinterpret_cast<int32*>(const_cast<char*>(split->tensor_data().data()));
  if (TF_PREDICT_FALSE(value == 0)) {
    return;
  }
  if (CHECK_EIGEN_ALIGN(sdata)) {
    split->flat<int32>() += split->flat<int32>().constant(value);
    return;
  }
  intptr_t sdata_ptr = reinterpret_cast<intptr_t>(sdata);
  size_t offset = EIGEN_MAX_ALIGN_BYTES - sdata_ptr % EIGEN_MAX_ALIGN_BYTES;
  offset /= sizeof(int32);
  const int32 ssize = split->NumElements();
  if (offset > ssize) {
    offset = ssize;
  }
  for (size_t i = 0; i < offset; ++i) {
    sdata[i] += value;
  }
  if (offset < ssize) {
    auto rslice = split->Slice(offset, ssize);
    rslice.flat<int32>() += rslice.flat<int32>().constant(value);
  }
}
}  // namespace anonymous

class RebatchTabularDatasetOp::Dataset::Iterator
    : public DatasetIterator<RebatchTabularDatasetOp::Dataset> {
 public:
  explicit Iterator(const Params& params)
      : DatasetIterator<RebatchTabularDatasetOp::Dataset>(params),
        queue_batch_size_(0) {
    tensor_queues_.resize(dataset()->field_ragged_indices_.size());
  }

  Status Initialize(IteratorContext* ctx) override {
    mutex_lock l(mu_);
    field_ranks_.clear();
    int64 prev_field_id = -1;
    int64 prev_ragged_index = -1;
    for (size_t i = 0; i < dataset()->field_ids_.size(); ++i) {
      const int64 field_id = dataset()->field_ids_[i];
      const int64 ragged_index = dataset()->field_ragged_indices_[i];
      auto output_dtype = dataset()->output_dtypes()[i];
      if (TF_PREDICT_FALSE(ragged_index > 1 && output_dtype != DT_INT32)) {
        return errors::InvalidArgument("Output tensor ", i,
                                       " must be DT_INT32 not ",
                                       DataTypeString(output_dtype));
      }
      if (ragged_index <= prev_ragged_index) {
        if (TF_PREDICT_FALSE(field_id == prev_field_id)) {
          return errors::Internal("Invalid `fields` serialization");
        }
        field_ranks_.push_back(prev_ragged_index);
      }
      prev_field_id = field_id;
      prev_ragged_index = ragged_index;
    }
    field_ranks_.push_back(prev_ragged_index);
    if (dataset()->num_parallel_scans_ > 1) {
#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) < 1015L
      thread_pool_.reset(new thread::ThreadPool(
          Env::Default(), ThreadOptions(),
          kRebatchTabularDatasetWorkerPool /* name */,
          dataset()->num_parallel_scans_ /* num_threads */,
          false /* low_latency_hint */));
#else
      thread_pool_ = ctx->CreateThreadPool(
          kRebatchTabularDatasetWorkerPool /* name */,
          dataset()->num_parallel_scans_ /* num_threads */);
#endif
    }
    return dataset()->input_dataset_->MakeIterator(ctx, prefix(), &input_impl_);
  }

  int64 GetBatchSize(const std::vector<Tensor>& input_tensors) {
    if (TF_PREDICT_FALSE(input_tensors.size() == 0)) {
      return 0;
    }
    if (field_ranks_[0] > 0) {
      return input_tensors[1].dim_size(0);
    }
    return input_tensors[0].dim_size(0);
  }

  Status Redirect(std::vector<Tensor>* output_tensors,
                  const std::vector<Tensor>& input_tensors) {
    *output_tensors = std::move(input_tensors);
    return Status::OK();
  }

  Status Redirect(std::vector<Tensor>* output_tensors,
                  const std::vector<Tensor>& input_tensors,
                  const int64 row_start, const int64 row_limit) {
    // Slice input tensors.
    int64 cur = 0;
    for (size_t fid = 0; fid < field_ranks_.size(); ++fid) {
      const int64 rank = field_ranks_[fid];
      if (rank == 0) {
        output_tensors->push_back(
            input_tensors[cur].Slice(row_start, row_limit));
        cur += (rank + 1);
        continue;
      }
      output_tensors->emplace_back();
      int64 start = row_start;
      int64 limit = row_limit;
      for (size_t i = 1; i < rank + 1; ++i) {
        auto s = input_tensors[cur + i].Slice(start, limit + 1);
        const int32* sdata =
            reinterpret_cast<int32*>(const_cast<char*>(s.tensor_data().data()));
        const int64 slimit = limit - start;
        start = sdata[0];
        limit = sdata[slimit];
        output_tensors->push_back(std::move(s));
      }
      output_tensors->at(cur) = input_tensors[cur].Slice(start, limit);
      cur += (rank + 1);
    }
    // Recalculate splits.
    auto recalc_splits = [this, &output_tensors](int64 rank, int64 cur) {
      for (size_t i = 1; i < rank + 1; ++i) {
        auto s = output_tensors->at(cur + i);
        const int32* sdata =
            reinterpret_cast<int32*>(const_cast<char*>(s.tensor_data().data()));
        RecalculateSplit(&s, -sdata[0]);
      }
    };
    if (thread_pool_) {
      BlockingCounter counter(field_ranks_.size());
      cur = 0;
      for (size_t fid = 0; fid < field_ranks_.size(); ++fid) {
        const int64 rank = field_ranks_[fid];
        if (rank == 0) {
          cur += (rank + 1);
          continue;
        }
        thread_pool_->Schedule([this, &recalc_splits, &counter, rank, cur]() {
          recalc_splits(rank, cur);
          counter.DecrementCount();
        });
        cur += (rank + 1);
      }
      counter.Wait();
    } else {
      cur = 0;
      for (size_t fid = 0; fid < field_ranks_.size(); ++fid) {
        const int64 rank = field_ranks_[fid];
        if (rank == 0) {
          cur += (rank + 1);
          continue;
        }
        recalc_splits(rank, cur);
        cur += (rank + 1);
      }
    }
    return Status::OK();
  }

  Status Enqueue(const std::vector<Tensor>& input_tensors,
                 const int64 batch_size) {
    if (TF_PREDICT_FALSE(batch_size == 0)) {
      return Status::OK();
    }
    for (size_t input_idx = 0; input_idx < input_tensors.size(); ++input_idx) {
      tensor_queues_[input_idx].push_back(input_tensors[input_idx]);
    }
    queue_batch_size_ += batch_size;
    return Status::OK();
  }

  Status Enqueue(const std::vector<Tensor>& input_tensors,
                 const int64 row_start, const int64 row_limit) {
    if (TF_PREDICT_FALSE(row_limit == row_start)) {
      return Status::OK();
    }
    int64 cur = 0;
    for (size_t fid = 0; fid < field_ranks_.size(); ++fid) {
      const int64 rank = field_ranks_[fid];
      if (rank == 0) {
        tensor_queues_[cur].push_back(
            input_tensors[cur].Slice(row_start, row_limit));
        cur += (rank + 1);
        continue;
      }
      int64 start = row_start;
      int64 limit = row_limit;
      for (size_t i = 1; i < rank + 1; ++i) {
        auto sliced = input_tensors[cur + i].Slice(start, limit + 1);
        int32* sdata = reinterpret_cast<int32*>(
            const_cast<char*>(sliced.tensor_data().data()));
        const int64 slimit = limit - start;
        start = sdata[0];
        limit = sdata[slimit];
        tensor_queues_[cur + i].push_back(std::move(sliced));
      }
      tensor_queues_[cur].push_back(input_tensors[cur].Slice(start, limit));
      cur += (rank + 1);
    }
    queue_batch_size_ += (row_limit - row_start);
    return Status::OK();
  }

  Status Dequeue(IteratorContext* ctx, std::vector<Tensor>* output_tensors) {
    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_on_host(true);
    Allocator* alloc = ctx->allocator(host_alloc_attrs);
    int64 cur = 0;
    for (size_t fid = 0; fid < field_ranks_.size(); ++fid) {
      // Create value tensor.
      auto output_dtype = dataset()->output_dtypes()[cur];
      PartialTensorShape output_pshape(dataset()->output_shapes_[cur]);
      if (output_pshape.dim_size(0) == -1) {
        int64 dim0_size = 0;
        for (auto& t : tensor_queues_[cur]) {
          dim0_size += t.dim_size(0);
        }
        output_pshape.set_dim(0, dim0_size);
      }
      TensorShape output_shape;
      output_pshape.AsTensorShape(&output_shape);
      if (TF_PREDICT_FALSE(output_dtype == DT_STRING)) {
        if (TF_PREDICT_FALSE(!TensorShapeUtils::IsVector(output_shape))) {
          return errors::InvalidArgument(
              "Tensors in string field must be vector");
        }
      }
      output_tensors->emplace_back(alloc, output_dtype, output_shape);

      const int64 rank = field_ranks_[fid];
      if (rank == 0) {
        cur += (rank + 1);
        continue;
      }

      // Create split tensors.
      for (size_t i = 1; i < rank + 1; ++i) {
        PartialTensorShape split_pshape(dataset()->output_shapes_[cur + i]);
        if (split_pshape.dim_size(0) == -1) {
          int64 dim0_size = 1;
          for (auto& t : tensor_queues_[cur + i]) {
            dim0_size += (t.dim_size(0) - 1);
          }
          split_pshape.set_dim(0, dim0_size);
        }
        TensorShape split_shape;
        split_pshape.AsTensorShape(&split_shape);
        output_tensors->emplace_back(alloc, DT_INT32, split_shape);
      }
      cur += (rank + 1);
    }

    auto flush_queue = [this, &output_tensors](int64 rank, int64 cur) {
      // Copy to value tensor.
      if (TF_PREDICT_FALSE(dataset()->output_dtypes()[cur] == DT_STRING)) {
        int output_cur = 0;
        for (auto& t : tensor_queues_[cur]) {
          const int tensor_size = t.NumElements();
          for (int i = 0; i < tensor_size; ++i) {
            output_tensors->at(cur).vec<string>()(output_cur + i) =
                t.unaligned_flat<string>()(i);
          }
          output_cur += tensor_size;
        }
      } else {
        char* output_ptr =
            const_cast<char*>(output_tensors->at(cur).tensor_data().data());
        for (auto& t : tensor_queues_[cur]) {
          const auto tensor_bytes = t.TotalBytes();
          assert(output_tensors->at(cur).TotalBytes() >= tensor_bytes);
          memcpy(output_ptr, t.tensor_data().data(), tensor_bytes);
          output_ptr += tensor_bytes;
        }
      }
      tensor_queues_[cur].clear();

      if (rank == 0) {
        return;
      }

      // Recalculate splits.
      for (size_t i = 1; i < rank + 1; ++i) {
        int32* sdata = reinterpret_cast<int32*>(const_cast<char*>(
            output_tensors->at(cur + i).tensor_data().data()));
        sdata[0] = 0;
        int32 tstart = 1;
        for (auto& t : tensor_queues_[cur + i]) {
          int32* tdata = reinterpret_cast<int32*>(
              const_cast<char*>(t.tensor_data().data()));
          memcpy(sdata + tstart, tdata + 1, t.TotalBytes() - sizeof(int32));
          const int32 tsize = t.NumElements() - 1;
          auto tpart =
              output_tensors->at(cur + i).Slice(tstart, tstart + tsize);
          RecalculateSplit(&tpart, sdata[tstart - 1] - tdata[0]);
          tstart += tsize;
        }
        tensor_queues_[cur + i].clear();
      }
    };

    if (thread_pool_) {
      BlockingCounter counter(field_ranks_.size());
      cur = 0;
      for (size_t fid = 0; fid < field_ranks_.size(); ++fid) {
        const int64 rank = field_ranks_[fid];
        thread_pool_->Schedule([this, &flush_queue, &counter, rank, cur]() {
          flush_queue(rank, cur);
          counter.DecrementCount();
        });
        cur += (rank + 1);
      }
      counter.Wait();
    } else {
      cur = 0;
      for (size_t fid = 0; fid < field_ranks_.size(); ++fid) {
        const int64 rank = field_ranks_[fid];
        flush_queue(rank, cur);
        cur += (rank + 1);
      }
    }

    queue_batch_size_ = 0;
    return Status::OK();
  }

  Status GetNextInternal(IteratorContext* ctx,
                         std::vector<Tensor>* output_tensors,
                         bool* end_of_sequence) override {
    mutex_lock l(mu_);
    *end_of_sequence = false;
    while (!*end_of_sequence) {
      std::vector<Tensor> input_tensors;
      int64 input_batch_size = 0;
      bool input_not_found = true;
      if (input_impl_) {
        // Current input is ready.
        bool input_end_of_sequence = false;
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, &input_tensors, &input_end_of_sequence));
        if (input_end_of_sequence) {
          input_impl_.reset();
        }
        input_batch_size = GetBatchSize(input_tensors);
        input_not_found = false;
      }
      int64 batch_size = queue_batch_size_ + input_batch_size;
      if (batch_size < dataset()->min_batch_size_) {
        // Merge batches.
        if (!input_impl_ && input_not_found) {
          // All inputs are read.
          if (TF_PREDICT_FALSE(batch_size == 0)) {
            *end_of_sequence = true;
            return Status::OK();
          }
          TF_RETURN_IF_ERROR(Dequeue(ctx, output_tensors));
          return Status::OK();
        } else {
          // Not all inputs are not read.
          TF_RETURN_IF_ERROR(Enqueue(input_tensors, input_batch_size));
          continue;
        }
      } else if (batch_size <= dataset()->batch_size_) {
        // Redirect or merge batches.
        if (queue_batch_size_ == 0) {
          // Queue is empty.
          TF_RETURN_IF_ERROR(Redirect(output_tensors, input_tensors));
          return Status::OK();
        } else if (!input_impl_ && input_not_found) {
          // Queue is not empty, all inputs are read.
          TF_RETURN_IF_ERROR(Dequeue(ctx, output_tensors));
          return Status::OK();
        } else {
          // Queue is not empty, not all inputs are consumed.
          TF_RETURN_IF_ERROR(Enqueue(input_tensors, input_batch_size));
          TF_RETURN_IF_ERROR(Dequeue(ctx, output_tensors));
          return Status::OK();
        }
      } else {
        // Split batches.
        if (queue_batch_size_ == 0) {
          // Queue is empty.
          TF_RETURN_IF_ERROR(Redirect(output_tensors, input_tensors, 0,
                                      dataset()->batch_size_));
          TF_RETURN_IF_ERROR(
              Enqueue(input_tensors, dataset()->batch_size_, input_batch_size));
          continue;
        } else {
          // Queue is not empty.
          const int64 residual_batch_size =
              dataset()->batch_size_ - queue_batch_size_;
          assert(residual_batch_size > 0);
          TF_RETURN_IF_ERROR(Enqueue(input_tensors, 0, residual_batch_size));
          TF_RETURN_IF_ERROR(Dequeue(ctx, output_tensors));
          TF_RETURN_IF_ERROR(
              Enqueue(input_tensors, residual_batch_size, input_batch_size));
          return Status::OK();
        }
      }
    }
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
  std::vector<int64> field_ranks_;
  std::vector<std::vector<Tensor>> tensor_queues_;
  int64 queue_batch_size_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

std::unique_ptr<IteratorBase>
RebatchTabularDatasetOp::Dataset::MakeIteratorInternal(
    const string& prefix) const {
  return std::unique_ptr<IteratorBase>(
      new Iterator({this, strings::StrCat(prefix, "::RebatchTabular")}));
}

REGISTER_KERNEL_BUILDER(Name("RebatchTabularDataset").Device(DEVICE_CPU),
                        RebatchTabularDatasetOp);

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("RebatchTabularDataset");

}  // namespace hybridbackend
}  // namespace tensorflow

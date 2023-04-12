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
#include <typeinfo>

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/partial_tensor_shape.h>
#include <tensorflow/core/framework/tensor.h>

#include "hybridbackend/tensorflow/common/dataset.h"

namespace tensorflow {
namespace hybridbackend {

enum TensorKinds {
  kSparseTensorIndices = 0,
  kTensorOrSparseTensorValues = 1,
  kSparseTensorDenseShape = 2
};

REGISTER_OP("HbSyncReplicasDataset")
    .Input("input_dataset: variant")
    .Attr("output_kinds: list(int) >= 1")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

class SyncReplicasDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit SyncReplicasDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_kinds", &output_kinds_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    *output = new Dataset(ctx, input, output_kinds_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const std::vector<int>& output_kinds)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          output_kinds_(output_kinds) {
      input_->Ref();
      output_dtypes_.emplace_back(DT_INT32);
      output_dtypes_.insert(output_dtypes_.end(),
                            input_->output_dtypes().begin(),
                            input_->output_dtypes().end());
      output_shapes_.emplace_back(TensorShape({}));
      output_shapes_.insert(output_shapes_.end(),
                            input_->output_shapes().begin(),
                            input_->output_shapes().end());
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, absl::StrCat(prefix, "::DetectEnd")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_dtypes_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    const std::vector<int>& output_kinds() const { return output_kinds_; }

    string DebugString() const override {
      return "SyncReplicasDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      AttrValue output_kinds;
      b->BuildAttrValue(output_kinds_, &output_kinds);
      TF_RETURN_IF_ERROR(b->AddDataset(this, {{0, input_graph_node}}, {},
                                       {{"output_kinds", output_kinds}},
                                       output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        std::vector<Tensor> buf;
        if (!buf_) {
          if (input_impl_) {
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, &buf, end_of_sequence));
          } else {
            *end_of_sequence = true;
          }
          if (*end_of_sequence) {
            input_impl_.reset();
            return Status::OK();
          } else {
            buf_.reset(new std::vector<Tensor>(std::move(buf)));
          }
        }
        if (input_impl_) {
          TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &buf, end_of_sequence));
        } else {
          *end_of_sequence = true;
        }

        out_tensors->emplace_back(Tensor(DT_INT32, {}));
        out_tensors->begin()->scalar<int32>()() = *end_of_sequence;
        if (!(*end_of_sequence) || input_impl_) {
          out_tensors->insert(out_tensors->end(), buf_->begin(), buf_->end());
        } else {
          std::vector<Tensor> dummy_out_tensors(buf_->size());
          for (int i = 0; i < dummy_out_tensors.size(); ++i) {
            if (dataset()->output_kinds()[i] == kSparseTensorDenseShape) {
              dummy_out_tensors[i] = Tensor(buf_->at(i));
              dummy_out_tensors[i].flat<int64>()(0) = 0;
            } else {
              TensorShape zeroed_shape;
              zeroed_shape.AppendShape(buf_->at(i).shape());
              zeroed_shape.set_dim(0, 0);
              dummy_out_tensors[i] = Tensor(buf_->at(i).dtype(), zeroed_shape);
            }
          }
          out_tensors->insert(out_tensors->end(), dummy_out_tensors.begin(),
                              dummy_out_tensors.end());
        }

        if (*end_of_sequence) {
          input_impl_.reset();
        } else {
          *buf_ = std::move(buf);
        }

        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        } else {
          TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (!reader->Contains(full_name("input_impl_empty"))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<std::vector<Tensor>> buf_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    DataTypeVector output_dtypes_;
    std::vector<PartialTensorShape> output_shapes_;
    const std::vector<int> output_kinds_;
  };

 private:
  std::vector<int> output_kinds_;
};

REGISTER_KERNEL_BUILDER(Name("HbSyncReplicasDataset").Device(DEVICE_CPU),
                        SyncReplicasDatasetOp);

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("HbSyncReplicasDataset");
}  // namespace hybridbackend
}  // namespace tensorflow

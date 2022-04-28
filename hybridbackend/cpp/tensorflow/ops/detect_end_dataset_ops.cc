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

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/partial_tensor_shape.h>
#include <tensorflow/core/framework/tensor.h>

#include "hybridbackend/cpp/tensorflow/io/dataset.h"

namespace tensorflow {
namespace hybridbackend {

REGISTER_OP("DetectEndDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

class DetectEndDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit DetectEndDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    *output = new Dataset(ctx, input);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input)
        : DatasetBase(DatasetContext(ctx)), input_(input) {
      input_->Ref();
      output_dtypes_.emplace_back(DT_BOOL);
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
          new Iterator({this, strings::StrCat(prefix, "::DetectEnd")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_dtypes_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "DetectEndDatasetOp()::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
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

        out_tensors->emplace_back(Tensor(DT_BOOL, {}));
        out_tensors->insert(out_tensors->end(), buf_->begin(), buf_->end());
        if (input_impl_) {
          TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &buf, end_of_sequence));
        } else {
          *end_of_sequence = true;
        }

        out_tensors->begin()->scalar<bool>()() = *end_of_sequence;
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
  };
};

REGISTER_KERNEL_BUILDER(Name("DetectEndDataset").Device(DEVICE_CPU),
                        DetectEndDatasetOp);

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("DetectEndDataset");
}  // namespace hybridbackend
}  // namespace tensorflow

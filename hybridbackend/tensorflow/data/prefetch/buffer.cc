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

#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <numeric>
#include <vector>

#if HYBRIDBACKEND_TENSORFLOW

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/framework/partial_tensor_shape.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/public/version.h>

namespace tensorflow {
namespace hybridbackend {

class PrefetchBuffer : public ResourceBase {
 public:
  explicit PrefetchBuffer(int64 capacity)
      : capacity_(capacity), is_cancelled_(false), is_closed_(false) {}

  ~PrefetchBuffer() { Cancel(); }

  Status Put(const std::vector<Tensor>& record) {
    std::unique_lock<std::mutex> lock(mu_);
    put_cv_.wait(
        lock, [this]() { return buffer_.size() < capacity_ || is_cancelled_; });

    if (TF_PREDICT_FALSE(is_cancelled_)) {
      lock.unlock();
      return Status(errors::Cancelled("Session was closed."));
    }

    buffer_.push_back(std::move(record));

    lock.unlock();
    take_cv_.notify_all();

    return Status::OK();
  }

  Status Take(std::vector<Tensor>* record) {
    std::unique_lock<std::mutex> lock(mu_);

    take_cv_.wait(lock, [this]() { return !buffer_.empty() || is_cancelled_; });

    if (TF_PREDICT_FALSE(is_closed_ && buffer_.empty())) {
      lock.unlock();
      return Status(errors::OutOfRange("EOF reached."));
    }

    if (TF_PREDICT_FALSE(is_cancelled_ && buffer_.empty())) {
      lock.unlock();
      return Status(errors::Cancelled("Session was closed."));
    }

    *record = std::move(buffer_.front());
    buffer_.pop_front();

    lock.unlock();
    put_cv_.notify_all();

    return Status::OK();
  }

  Status Cancel(bool is_cancelled = true) {
    std::unique_lock<std::mutex> lock(mu_);

    is_cancelled_ = is_cancelled;

    lock.unlock();
    put_cv_.notify_all();
    take_cv_.notify_all();
    return Status::OK();
  }

  Status Close() {
    std::unique_lock<std::mutex> lock(mu_);

    is_cancelled_ = true;
    is_closed_ = true;

    lock.unlock();
    put_cv_.notify_all();
    take_cv_.notify_all();
    return Status::OK();
  }

  Status GetSize(Tensor* size) {
    std::unique_lock<std::mutex> lock(mu_);
    size->scalar<int32>().setConstant(static_cast<int64>(buffer_.size()));
    return Status::OK();
  }

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1015L
  string DebugString() const override {
    return strings::StrCat("PrefetchBuffer(capacity=", capacity_, ")");
  }
#else
  string DebugString() override {
    return strings::StrCat("PrefetchBuffer(capacity=", capacity_, ")");
  }
#endif

  void Schedule(const string& name, int64 num_threads,
                std::function<void()> fn) {
    std::unique_lock<std::mutex> lock(mu_);
    if (threads_) {
      lock.unlock();
      threads_->Schedule(fn);
      return;
    }

    threads_.reset(
        new thread::ThreadPool(Env::Default(), ThreadOptions(),
                               strings::StrCat("data_buffer_threads_", name),
                               num_threads, false /* low_latency_hint */));

    lock.unlock();
    threads_->Schedule(fn);
  }

 private:
  std::deque<std::vector<Tensor> > buffer_;
  std::size_t capacity_;
  bool is_cancelled_;
  bool is_closed_;
  std::mutex mu_;
  std::condition_variable take_cv_;
  std::condition_variable put_cv_;
  std::shared_ptr<thread::ThreadPool> threads_;
};

class PrefetchBufferOp : public OpKernel {
 public:
  explicit PrefetchBufferOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto rm = ctx->resource_manager();
    auto ndef = def();

    ContainerInfo cinfo;
    OP_REQUIRES_OK(ctx, cinfo.Init(rm, ndef, true /* use name() */));

    PrefetchBuffer* buffer = nullptr;
    OP_REQUIRES_OK(ctx, rm->LookupOrCreate<PrefetchBuffer>(
                            cinfo.container(), cinfo.name(), &buffer,
                            [&ndef](PrefetchBuffer** pbuf) -> Status {
                              int64 capacity;
                              TF_RETURN_IF_ERROR(GetNodeAttr(
                                  ndef, "shared_capacity", &capacity));
                              *pbuf = new PrefetchBuffer(capacity);
                              return Status::OK();
                            }));
    core::ScopedUnref scope(buffer);
    ComputeWithPrefetchBuffer(ctx, buffer);
  }

 protected:
  virtual void ComputeWithPrefetchBuffer(OpKernelContext* ctx,
                                         PrefetchBuffer* buf) = 0;
};

class PrefetchBufferAsyncOp : public AsyncOpKernel {
 public:
  explicit PrefetchBufferAsyncOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_threads", &shared_threads_));
  }

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    auto rm = ctx->resource_manager();
    NodeDef ndef(def());
    ContainerInfo cinfo;
    OP_REQUIRES_OK_ASYNC(ctx, cinfo.Init(rm, ndef, true /* use name() */),
                         done);
    PrefetchBuffer* buffer = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx,
                         rm->LookupOrCreate<PrefetchBuffer>(
                             cinfo.container(), cinfo.name(), &buffer,
                             [&ndef](PrefetchBuffer** resource) {
                               int64 capacity;
                               TF_RETURN_IF_ERROR(GetNodeAttr(
                                   ndef, "shared_capacity", &capacity));
                               *resource = new PrefetchBuffer(capacity);
                               return Status::OK();
                             }),
                         done);
    core::ScopedUnref scoped_list(buffer);
    Schedule(buffer, [this, ctx, done, buffer]() {
      ComputeAsyncWithPrefetchBuffer(ctx, done, buffer);
    });
  }

 protected:
  virtual void ComputeAsyncWithPrefetchBuffer(OpKernelContext* ctx,
                                              AsyncOpKernel::DoneCallback done,
                                              PrefetchBuffer* buffer) = 0;

 private:
  string shared_name_;
  int64 shared_threads_;

  void Schedule(PrefetchBuffer* buffer, std::function<void()> fn) {
    buffer->Schedule(shared_name_, shared_threads_, fn);
  }
};

REGISTER_OP("HbPrefetchBufferPut")
    .Input("record: dtypes")
    .Attr("container: string = ''")
    .Attr("dtypes: list(type)")
    .Attr("shared_name: string = ''")
    .Attr("shared_capacity: int >= 1 = 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

class PrefetchBufferPutOp : public PrefetchBufferOp {
 public:
  explicit PrefetchBufferPutOp(OpKernelConstruction* ctx)
      : PrefetchBufferOp(ctx) {}

  void ComputeWithPrefetchBuffer(OpKernelContext* ctx,
                                 PrefetchBuffer* buf) override {
    std::vector<Tensor> record;
    record.reserve(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      record.push_back(ctx->input(i));
    }
    ctx->SetStatus(buf->Put(record));
  }
};

REGISTER_KERNEL_BUILDER(Name("HbPrefetchBufferPut").Device(DEVICE_CPU),
                        PrefetchBufferPutOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbPrefetchBufferPut").Device(DEVICE_GPU),
                        PrefetchBufferPutOp);
#endif  // GOOGLE_CUDA

REGISTER_OP("HbPrefetchBufferTake")
    .Output("record: dtypes")
    .Attr("container: string = ''")
    .Attr("dtypes: list(type)")
    .Attr("shared_name: string = ''")
    .Attr("shared_capacity: int >= 1 = 1")
    .Attr("shared_threads: int >= 1 = 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

class PrefetchBufferTakeOp : public PrefetchBufferAsyncOp {
 public:
  explicit PrefetchBufferTakeOp(OpKernelConstruction* ctx)
      : PrefetchBufferAsyncOp(ctx) {}

  void ComputeAsyncWithPrefetchBuffer(OpKernelContext* ctx,
                                      AsyncOpKernel::DoneCallback done,
                                      PrefetchBuffer* buf) override {
    std::vector<Tensor> record;
    Status s = buf->Take(&record);
    if (TF_PREDICT_FALSE(!s.ok())) {
      ctx->SetStatus(s);
      done();
      return;
    }

    OP_REQUIRES_ASYNC(
        ctx, record.size() == (size_t)ctx->num_outputs(),
        errors::Internal(ctx->num_outputs(), " tensors required, but ",
                         record.size(), " tensors were taken."),
        done);

    for (size_t i = 0; i < record.size(); ++i) {
      ctx->set_output(i, record[i]);
    }
    done();
  }
};

REGISTER_KERNEL_BUILDER(Name("HbPrefetchBufferTake").Device(DEVICE_CPU),
                        PrefetchBufferTakeOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbPrefetchBufferTake").Device(DEVICE_GPU),
                        PrefetchBufferTakeOp);
#endif  // GOOGLE_CUDA

REGISTER_OP("HbPrefetchBufferCancel")
    .Attr("container: string = ''")
    .Attr("is_cancelled: bool = true")
    .Attr("shared_name: string = ''")
    .Attr("shared_capacity: int >= 1 = 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

class PrefetchBufferCancelOp : public PrefetchBufferOp {
 public:
  explicit PrefetchBufferCancelOp(OpKernelConstruction* ctx)
      : PrefetchBufferOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_cancelled", &is_cancelled_));
  }

  void ComputeWithPrefetchBuffer(OpKernelContext* ctx,
                                 PrefetchBuffer* buf) override {
    ctx->SetStatus(buf->Cancel(is_cancelled_));
  }

 private:
  bool is_cancelled_;
};

REGISTER_KERNEL_BUILDER(Name("HbPrefetchBufferCancel").Device(DEVICE_CPU),
                        PrefetchBufferCancelOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbPrefetchBufferCancel").Device(DEVICE_GPU),
                        PrefetchBufferCancelOp);
#endif  // GOOGLE_CUDA

REGISTER_OP("HbPrefetchBufferClose")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("shared_capacity: int >= 1 = 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

class PrefetchBufferCloseOp : public PrefetchBufferOp {
 public:
  explicit PrefetchBufferCloseOp(OpKernelConstruction* ctx)
      : PrefetchBufferOp(ctx) {}

  void ComputeWithPrefetchBuffer(OpKernelContext* ctx,
                                 PrefetchBuffer* buf) override {
    ctx->SetStatus(buf->Close());
  }
};

REGISTER_KERNEL_BUILDER(Name("HbPrefetchBufferClose").Device(DEVICE_CPU),
                        PrefetchBufferCloseOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbPrefetchBufferClose").Device(DEVICE_GPU),
                        PrefetchBufferCloseOp);
#endif  // GOOGLE_CUDA

REGISTER_OP("HbPrefetchBufferSize")
    .Output("size: int32")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("shared_capacity: int >= 1 = 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .SetIsStateful();

class PrefetchBufferSizeOp : public PrefetchBufferOp {
 public:
  explicit PrefetchBufferSizeOp(OpKernelConstruction* ctx)
      : PrefetchBufferOp(ctx) {}

  void ComputeWithPrefetchBuffer(OpKernelContext* ctx,
                                 PrefetchBuffer* buf) override {
    Tensor* size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &size));
    OP_REQUIRES_OK(ctx, buf->GetSize(size));
  }
};

REGISTER_KERNEL_BUILDER(Name("HbPrefetchBufferSize").Device(DEVICE_CPU),
                        PrefetchBufferSizeOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("HbPrefetchBufferSize").HostMemory("size").Device(DEVICE_GPU),
    PrefetchBufferSizeOp);
#endif  // GOOGLE_CUDA

#endif  // HYBRIDBACKEND_TENSORFLOW

}  // namespace hybridbackend
}  // namespace tensorflow
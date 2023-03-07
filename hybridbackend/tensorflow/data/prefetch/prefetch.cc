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

#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <numeric>
#include <vector>

#if HYBRIDBACKEND_TENSORFLOW

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/node_def.pb.h>
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
  explicit PrefetchBuffer(const string& name, const int64 capacity,
                          const int64 num_takers, const int64 num_runners)
      : name_(name),
        capacity_(capacity),
        num_takers_(num_takers),
        num_runners_(num_runners),
        is_cancelled_(false),
        is_closed_(false) {}

  ~PrefetchBuffer() { Cancel(); }

  Status Take(std::vector<Tensor>* values) {
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

    *values = std::move(buffer_.front());
    buffer_.pop_front();

    lock.unlock();
    put_cv_.notify_all();

    return Status::OK();
  }

  Status Put(const std::vector<Tensor>& values) {
    std::unique_lock<std::mutex> lock(mu_);
    put_cv_.wait(
        lock, [this]() { return buffer_.size() < capacity_ || is_cancelled_; });

    if (TF_PREDICT_FALSE(is_cancelled_)) {
      lock.unlock();
      return Status(errors::Cancelled("Session was closed."));
    }

    buffer_.push_back(std::move(values));

    lock.unlock();
    take_cv_.notify_all();

    return Status::OK();
  }

  Status Cancel() {
    std::unique_lock<std::mutex> lock(mu_);

    is_cancelled_ = true;

    lock.unlock();
    put_cv_.notify_all();
    take_cv_.notify_all();
    return Status::OK();
  }

  Status Resume() {
    std::unique_lock<std::mutex> lock(mu_);

    is_cancelled_ = false;

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

  void TakeAsync(OpKernelContext* ctx, AsyncOpKernel::DoneCallback done) {
    Schedule("takers", num_takers_, &takers_, [this, ctx, done]() {
      std::vector<Tensor> values;
      Status s = this->Take(&values);
      if (TF_PREDICT_FALSE(!s.ok())) {
        ctx->SetStatus(s);
        done();
        return;
      }

      OP_REQUIRES_ASYNC(
          ctx, values.size() == (size_t)ctx->num_outputs(),
          errors::Internal(ctx->num_outputs(), " tensors required, but ",
                           values.size(), " tensors were taken."),
          done);

      for (size_t i = 0; i < values.size(); ++i) {
        ctx->set_output(i, std::move(values[i]));
      }
      done();
    });
  }

  void PutAsync(OpKernelContext* ctx, AsyncOpKernel::DoneCallback done) {
    Schedule("runners", num_runners_, &runners_, [this, ctx, done]() {
      std::vector<Tensor> values;
      values.reserve(ctx->num_inputs());
      for (int i = 0; i < ctx->num_inputs(); ++i) {
        values.emplace_back(ctx->input(i));
      }
      ctx->SetStatus(Put(std::move(values)));
      done();
    });
  }

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1015L
  string DebugString() const override {
    return absl::StrCat("PrefetchBuffer(name=", name_, ", capacity=", capacity_,
                        ", num_takers=", num_takers_,
                        ", num_runners=", num_runners_, ")");
  }
#else
  string DebugString() override {
    return absl::StrCat("PrefetchBuffer(name=", name_, ", capacity=", capacity_,
                        ", num_takers=", num_takers_,
                        ", num_runners=", num_runners_, ")");
  }
#endif

 private:
  void Schedule(const string& tag, const int64 num_threads,
                std::shared_ptr<thread::ThreadPool>* threads,
                std::function<void()> fn) {
    std::unique_lock<std::mutex> lock(mu_);
    if (*threads) {
      lock.unlock();
      (*threads)->Schedule(fn);
      return;
    }

    threads->reset(
        new thread::ThreadPool(Env::Default(), ThreadOptions(),
                               absl::StrCat("data_buffer_", name_, "_", tag),
                               num_threads, false /* low_latency_hint */));

    lock.unlock();
    (*threads)->Schedule(fn);
  }

  string name_;
  std::deque<std::vector<Tensor>> buffer_;
  int64 capacity_;
  int64 num_takers_;
  int64 num_runners_;
  bool is_cancelled_;
  bool is_closed_;
  std::mutex mu_;
  std::condition_variable take_cv_;
  std::condition_variable put_cv_;
  std::shared_ptr<thread::ThreadPool> takers_;
  std::shared_ptr<thread::ThreadPool> runners_;
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
    OP_REQUIRES_OK(
        ctx,
        rm->LookupOrCreate<PrefetchBuffer>(
            cinfo.container(), cinfo.name(), &buffer,
            [&ndef](PrefetchBuffer** pbuf) -> Status {
              string shared_name;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(ndef, "shared_name", &shared_name));
              int64 capacity;
              TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "capacity", &capacity));
              int64 num_takers;
              TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "num_takers", &num_takers));
              int64 num_runners;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(ndef, "num_runners", &num_runners));
              *pbuf = new PrefetchBuffer(shared_name, capacity, num_takers,
                                         num_runners);
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
      : AsyncOpKernel(ctx) {}

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    auto rm = ctx->resource_manager();
    NodeDef ndef(def());
    ContainerInfo cinfo;
    OP_REQUIRES_OK_ASYNC(ctx, cinfo.Init(rm, ndef, true /* use name() */),
                         done);
    PrefetchBuffer* buffer = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx,
        rm->LookupOrCreate<PrefetchBuffer>(
            cinfo.container(), cinfo.name(), &buffer,
            [&ndef](PrefetchBuffer** pbuf) {
              string shared_name;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(ndef, "shared_name", &shared_name));
              int64 capacity;
              TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "capacity", &capacity));
              int64 num_takers;
              TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "num_takers", &num_takers));
              int64 num_runners;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(ndef, "num_runners", &num_runners));
              *pbuf = new PrefetchBuffer(shared_name, capacity, num_takers,
                                         num_runners);
              return Status::OK();
            }),
        done);
    core::ScopedUnref scoped_list(buffer);
    ComputeAsyncWithPrefetchBuffer(ctx, buffer, done);
  }

 protected:
  virtual void ComputeAsyncWithPrefetchBuffer(
      OpKernelContext* ctx, PrefetchBuffer* buffer,
      AsyncOpKernel::DoneCallback done) = 0;
};

REGISTER_OP("HbTakeFromPrefetch")
    .Output("values: dtypes")
    .Attr("num_takers: int >= 1 = 1")
    .Attr("num_runners: int >= 1 = 1")
    .Attr("capacity: int >= 1 = 1")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

class HbTakeFromPrefetchOp : public PrefetchBufferAsyncOp {
 public:
  explicit HbTakeFromPrefetchOp(OpKernelConstruction* ctx)
      : PrefetchBufferAsyncOp(ctx) {}

  void ComputeAsyncWithPrefetchBuffer(
      OpKernelContext* ctx, PrefetchBuffer* buffer,
      AsyncOpKernel::DoneCallback done) override {
    buffer->TakeAsync(ctx, done);
  }
};

REGISTER_KERNEL_BUILDER(Name("HbTakeFromPrefetch").Device(DEVICE_CPU),
                        HbTakeFromPrefetchOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbTakeFromPrefetch").Device(DEVICE_GPU),
                        HbTakeFromPrefetchOp);
#endif  // GOOGLE_CUDA

REGISTER_OP("HbRunPrefetch")
    .Input("values: dtypes")
    .Attr("num_takers: int >= 1 = 1")
    .Attr("num_runners: int >= 1 = 1")
    .Attr("capacity: int >= 1 = 1")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

class RunPrefetchOp : public PrefetchBufferAsyncOp {
 public:
  explicit RunPrefetchOp(OpKernelConstruction* ctx)
      : PrefetchBufferAsyncOp(ctx) {}

  void ComputeAsyncWithPrefetchBuffer(
      OpKernelContext* ctx, PrefetchBuffer* buffer,
      AsyncOpKernel::DoneCallback done) override {
    buffer->PutAsync(ctx, done);
  }
};

REGISTER_KERNEL_BUILDER(Name("HbRunPrefetch").Device(DEVICE_CPU),
                        RunPrefetchOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbRunPrefetch").Device(DEVICE_GPU),
                        RunPrefetchOp);
#endif  // GOOGLE_CUDA

REGISTER_OP("HbCancelPrefetch")
    .Attr("num_takers: int >= 1 = 1")
    .Attr("num_runners: int >= 1 = 1")
    .Attr("capacity: int >= 1 = 1")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

class CancelPrefetchOp : public PrefetchBufferOp {
 public:
  explicit CancelPrefetchOp(OpKernelConstruction* ctx)
      : PrefetchBufferOp(ctx) {}

  void ComputeWithPrefetchBuffer(OpKernelContext* ctx,
                                 PrefetchBuffer* buf) override {
    ctx->SetStatus(buf->Cancel());
  }
};

REGISTER_KERNEL_BUILDER(Name("HbCancelPrefetch").Device(DEVICE_CPU),
                        CancelPrefetchOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbCancelPrefetch").Device(DEVICE_GPU),
                        CancelPrefetchOp);
#endif  // GOOGLE_CUDA

REGISTER_OP("HbResumePrefetch")
    .Attr("num_takers: int >= 1 = 1")
    .Attr("num_runners: int >= 1 = 1")
    .Attr("capacity: int >= 1 = 1")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

class ResumePrefetchOp : public PrefetchBufferOp {
 public:
  explicit ResumePrefetchOp(OpKernelConstruction* ctx)
      : PrefetchBufferOp(ctx) {}

  void ComputeWithPrefetchBuffer(OpKernelContext* ctx,
                                 PrefetchBuffer* buf) override {
    ctx->SetStatus(buf->Resume());
  }
};

REGISTER_KERNEL_BUILDER(Name("HbResumePrefetch").Device(DEVICE_CPU),
                        ResumePrefetchOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbResumePrefetch").Device(DEVICE_GPU),
                        ResumePrefetchOp);
#endif  // GOOGLE_CUDA

REGISTER_OP("HbStopPrefetch")
    .Attr("num_takers: int >= 1 = 1")
    .Attr("num_runners: int >= 1 = 1")
    .Attr("capacity: int >= 1 = 1")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

class StopPrefetchOp : public PrefetchBufferOp {
 public:
  explicit StopPrefetchOp(OpKernelConstruction* ctx) : PrefetchBufferOp(ctx) {}

  void ComputeWithPrefetchBuffer(OpKernelContext* ctx,
                                 PrefetchBuffer* buf) override {
    ctx->SetStatus(buf->Close());
  }
};

REGISTER_KERNEL_BUILDER(Name("HbStopPrefetch").Device(DEVICE_CPU),
                        StopPrefetchOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("HbStopPrefetch").Device(DEVICE_GPU),
                        StopPrefetchOp);
#endif  // GOOGLE_CUDA

REGISTER_OP("HbPrefetchBufferSize")
    .Output("size: int32")
    .Attr("num_takers: int >= 1 = 1")
    .Attr("num_runners: int >= 1 = 1")
    .Attr("capacity: int >= 1 = 1")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
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
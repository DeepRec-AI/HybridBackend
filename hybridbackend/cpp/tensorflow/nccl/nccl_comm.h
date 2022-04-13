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

#ifndef HYBRIDBACKEND_CPP_TENSORFLOW_NCCL_NCCL_COMM_H_
#define HYBRIDBACKEND_CPP_TENSORFLOW_NCCL_NCCL_COMM_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/public/version.h>

#if GOOGLE_CUDA
#include <tensorflow/core/common_runtime/gpu/gpu_event_mgr.h>
#include <tensorflow/stream_executor/stream_executor.h>
#endif

#include "hybridbackend/cpp/tensorflow/nccl/nccl.h"

namespace tensorflow {

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1008L
namespace se = ::perftools::gputools;
#endif

namespace hybridbackend {

#if HYBRIDBACKEND_NCCL
#if GOOGLE_CUDA

class NcclComm : public ResourceBase {
 public:
  int rank() const { return rank_; }

  int size() const { return size_; }

  se::Stream* ctx_stream() const { return ctx_stream_; }
  cudaStream_t* stream() const { return stream_; }

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1014L
  string DebugString() const override {
#else
  string DebugString() override {
#endif
    return debug_string_;
  }

  NcclComm();

  ~NcclComm();

  Status Initialize(int size, int rank, const string& shared_name,
                    OpKernelContext* ctx);

  Status Create(const string& id);

  Status Destroy();

  Status Abort();

  Status UserRank(int* rank);

  void RunAsync(std::function<void()> func, OpKernelContext* ctx,
                AsyncOpKernel::DoneCallback done);

  se::Event* ThenRecordEvent();

  void ThenWaitFor(se::Event* ev);

  void BlockHostUntilDone();

  // All-to-one computation

  Status Reduce(const Tensor& input, const ncclRedOp_t reduce_op,
                const int root_rank, Tensor* output);

  // All-to-all computation

  Status ReduceScatter(const Tensor& input, const ncclRedOp_t reduce_op,
                       Tensor* output);

  Status Allreduce(const Tensor& input, const ncclRedOp_t reduce_op,
                   Tensor* output);

  // One-to-all data movement

  Status Broadcast(const Tensor& input, const int root_rank, Tensor* output);

  // Scatter not supported.

  // All-to-one data movement

  // Gather not supported.
  // Gatherv not supported.

  // All-to-all data movement

  Status Allgather(const Tensor& input, Tensor* output);

  Status GroupAllgather(const std::vector<Tensor>& inputs, Tensor* output);

  Status Allgatherv(const Tensor& input, const Tensor& host_sizes,
                    Tensor* output);

  Status Alltoall(const Tensor& input, Tensor* output);

  Status Alltoallv(const Tensor& input, const Tensor& host_sizes,
                   const int64 common_size, Tensor* output);

  Status GroupAlltoallv(const std::vector<Tensor>& inputs,
                        const Tensor& host_sizes,
                        const std::vector<int64>& common_sizes,
                        std::vector<Tensor*>* outputs);

  Status GroupAlltoallv(const std::vector<Tensor*>& inputs,
                        const Tensor& host_sizes,
                        const std::vector<int64>& common_sizes,
                        std::vector<Tensor*>* outputs);

  Status Alltoallw(const std::vector<Tensor>& inputs,
                   std::vector<Tensor*>* outputs);

  Status Alltoallw(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>* outputs);

  Status GroupAlltoallw(const std::vector<Tensor>& inputs,
                        const int64 group_size, std::vector<Tensor*>* outputs);

  Status GroupAlltoallw(const std::vector<Tensor*>& inputs,
                        const int64 group_size, std::vector<Tensor*>* outputs);

 private:
  ncclComm_t comm_;
  int size_;
  int rank_;
  bool created_;
  string debug_string_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  se::Stream* ctx_stream_;
  cudaStream_t* stream_;
};

class NcclCommAsyncOp : public AsyncOpKernel {
 public:
  explicit NcclCommAsyncOp(OpKernelConstruction* ctx);

  virtual void ComputeAsyncWithComm(NcclComm* comm, OpKernelContext* ctx,
                                    DoneCallback done) = 0;

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override;

  se::Event* ThenRecordEvent(OpKernelContext* ctx);

  void ThenWaitFor(OpKernelContext* ctx, se::Event* ev);

  void ThenExecute(OpKernelContext* ctx, std::function<void()> func);

  void ThenCopyToDevice(OpKernelContext* ctx, Tensor* dst, const Tensor& src);

  void ThenCopyToHost(OpKernelContext* ctx, Tensor* dst, const Tensor& src);

  void BlockHostUntilDone(OpKernelContext* ctx);
};

#endif
#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif

#endif  // HYBRIDBACKEND_CPP_TENSORFLOW_NCCL_NCCL_COMM_H_

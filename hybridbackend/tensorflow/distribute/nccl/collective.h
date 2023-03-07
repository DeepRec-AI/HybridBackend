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

#ifndef HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_NCCL_COLLECTIVE_H_
#define HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_NCCL_COLLECTIVE_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/public/version.h>

#if GOOGLE_CUDA
#include <tensorflow/core/common_runtime/gpu/gpu_event_mgr.h>
#include <tensorflow/stream_executor/stream_executor.h>
#endif

#include "hybridbackend/tensorflow/common/stream.h"
#include "hybridbackend/tensorflow/distribute/collective.h"
#include "hybridbackend/tensorflow/distribute/nccl/types.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL
#if GOOGLE_CUDA

class NcclCollective : public Collective {
 public:
  Stream* stream() const { return stream_; }

  // Lifecycle management
  NcclCollective(const string& shared_name, int world_size, int local_size,
                 int rank);

  virtual Status Initialize(OpKernelContext* ctx) override;

  virtual Status Destroy() override;

  // Symmetric collective operations

  virtual Status Allreduce(const Tensor& input,
                           const CollectiveReduceOp reduce_op,
                           Tensor* output) override;

  virtual Status AllreduceN(const std::vector<Tensor>& n_input,
                            const CollectiveReduceOp reduce_op,
                            std::vector<Tensor*>* n_output) override;

  virtual Status Alltoall(const Tensor& input, Tensor* output,
                          CollectiveTopology topology) override;

  virtual Status AlltoallN(const std::vector<Tensor>& n_input,
                           std::vector<Tensor*>* n_output,
                           CollectiveTopology topology) override;

  virtual Status AlltoallN(const std::vector<Tensor*>& n_input,
                           std::vector<Tensor*>* n_output,
                           CollectiveTopology topology) override;

  virtual Status Alltoallv(const Tensor& input, const int32* send_sizes,
                           const int32* recv_sizes, const int64 common_size,
                           Tensor* output,
                           CollectiveTopology topology) override;

  virtual Status AlltoallvN(const std::vector<Tensor>& n_input,
                            const std::vector<int32*>& n_send_sizes,
                            const std::vector<int32*>& n_recv_sizes,
                            const std::vector<int64>& n_common_size,
                            std::vector<Tensor*>* n_output,
                            CollectiveTopology topology) override;

  virtual Status AlltoallvN(const std::vector<Tensor*>& n_input,
                            const std::vector<int32*>& n_send_sizes,
                            const std::vector<int32*>& n_recv_sizes,
                            const std::vector<int64>& n_common_size,
                            std::vector<Tensor*>* n_output,
                            CollectiveTopology topology) override;

  // Asymmetric collective operations

  virtual Status Broadcast(const Tensor& input, const int root_rank,
                           Tensor* output) override;

  virtual Status Allgather(const Tensor& input, Tensor* output) override;

  virtual Status Allgatherv(const Tensor& input, const Tensor& host_sizes,
                            Tensor* output) override;

  // NCCL specific operations

  virtual Status Create(const string& id);

  virtual Status CheckAsyncErrors();

 private:
  ncclComm_t comm_;
  bool created_;
  Stream* stream_;
  std::mutex mu_;
};

class NcclCollectiveAsyncOp : public AsyncOpKernel {
 public:
  explicit NcclCollectiveAsyncOp(OpKernelConstruction* ctx);

  virtual void CollectiveComputeAsync(NcclCollective* coll,
                                      OpKernelContext* ctx,
                                      DoneCallback done) = 0;

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override;
};

#endif
#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif

#endif  // HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_NCCL_COLLECTIVE_H_

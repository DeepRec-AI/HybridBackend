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

#ifndef HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_COLLECTIVE_H_
#define HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_COLLECTIVE_H_

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

namespace tensorflow {

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1008L
namespace se = ::perftools::gputools;
#endif

namespace hybridbackend {

#if HYBRIDBACKEND_NCCL
#if GOOGLE_CUDA

enum CollectiveReduceOp {
  kCollectiveSum = 0,
  kCollectiveProd = 1,
  kCollectiveMax = 2,
  kCollectiveMin = 3,
  kCollectiveAvg = 4
};

enum CollectiveTopology {
  kCollectiveAllDevices = 0,
  kCollectiveDevicesIntraNode = 1,
  kCollectiveDevicesInterNode = 2
};

inline string CollectiveTopologyString(const CollectiveTopology& topology) {
  switch (topology) {
    case kCollectiveAllDevices:
      return "AllDevices";
    case kCollectiveDevicesIntraNode:
      return "DevicesIntraNode";
    case kCollectiveDevicesInterNode:
      return "DevicesInterNode";
  }
  return "Unknown";
}

class Collective : public ResourceBase {
 public:
  string shared_name() const { return shared_name_; }

  int world_size() const { return world_size_; }

  int local_size() const { return local_size_; }

  int rank() const { return rank_; }

  void compute_active_ranks(CollectiveTopology topology,
                            std::vector<int>& out) {
    if (topology == kCollectiveDevicesIntraNode) {
      int node_idx = rank_ / local_size_;
      for (int rank = node_idx * local_size_;
           rank < (node_idx + 1) * local_size_; ++rank) {
        out.emplace_back(rank);
      }
    } else if (topology == kCollectiveDevicesInterNode) {
      for (int rank = 0; rank < world_size_; ++rank) {
        if (local_size_ == 1 || (rank % local_size_) == (rank_ % local_size_)) {
          out.emplace_back(rank);
        }
      }
    } else {
      for (int rank = 0; rank < world_size_; ++rank) {
        out.emplace_back(rank);
      }
    }
  }

  int compute_active_size(CollectiveTopology topology) {
    int active_size;
    if (topology == kCollectiveDevicesIntraNode) {
      active_size = local_size_;
    } else if (topology == kCollectiveDevicesInterNode) {
      active_size = world_size_ / local_size_;
    } else {
      active_size = world_size_;
    }

    return active_size;
  }

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1014L
  string DebugString() const override {
#else
  string DebugString() override {
#endif
    return absl::StrCat("Collective(shared_name=", shared_name_,
                        ", world_size=", world_size_,
                        ", local_size=", local_size_, ", rank=", rank_, ")");
  }

  // Lifecycle management

  Collective(const string& shared_name, int world_size, int local_size,
             int rank)
      : shared_name_(shared_name),
        world_size_(world_size),
        local_size_(local_size),
        rank_(rank) {}

  virtual ~Collective() { Destroy(); }

  virtual Status Initialize(OpKernelContext* ctx) {
    if (!TF_PREDICT_TRUE(0 <= rank_ && rank_ < world_size_)) {
      return errors::InvalidArgument(absl::StrCat("Collective rank ", rank_,
                                                  " or world_size ",
                                                  world_size_, " is invalid."));
    }

    return Status::OK();
  };

  virtual Status Destroy() { return Status::OK(); };

  // Symmetric collective operations

  virtual Status Allreduce(const Tensor& input,
                           const CollectiveReduceOp reduce_op,
                           Tensor* output) = 0;

  virtual Status AllreduceN(const std::vector<Tensor>& n_input,
                            const CollectiveReduceOp reduce_op,
                            std::vector<Tensor*>* n_output) = 0;

  virtual Status Alltoall(const Tensor& input, Tensor* output,
                          CollectiveTopology topology) = 0;

  virtual Status AlltoallN(const std::vector<Tensor>& n_input,
                           std::vector<Tensor*>* n_output,
                           CollectiveTopology topology) = 0;

  virtual Status AlltoallN(const std::vector<Tensor*>& n_input,
                           std::vector<Tensor*>* n_output,
                           CollectiveTopology topology) = 0;

  virtual Status Alltoallv(const Tensor& input, const int32* send_sizes,
                           const int32* recv_sizes, const int64 common_size,
                           Tensor* output, CollectiveTopology topology) = 0;

  virtual Status AlltoallvN(const std::vector<Tensor>& n_input,
                            const std::vector<int32*>& n_send_sizes,
                            const std::vector<int32*>& n_recv_sizes,
                            const std::vector<int64>& n_common_size,
                            std::vector<Tensor*>* n_output,
                            CollectiveTopology topology) = 0;

  virtual Status AlltoallvN(const std::vector<Tensor*>& n_input,
                            const std::vector<int32*>& n_send_sizes,
                            const std::vector<int32*>& n_recv_sizes,
                            const std::vector<int64>& n_common_size,
                            std::vector<Tensor*>* n_output,
                            CollectiveTopology topology) = 0;

  // Asymmetric collective operations

  virtual Status Broadcast(const Tensor& input, const int root_rank,
                           Tensor* output) = 0;

  virtual Status Allgather(const Tensor& input, Tensor* output) = 0;

  virtual Status Allgatherv(const Tensor& input, const Tensor& host_sizes,
                            Tensor* output) = 0;

 private:
  string shared_name_;
  int world_size_;
  int local_size_;
  int rank_;
};

#endif
#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif

#endif  // HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_COLLECTIVE_H_

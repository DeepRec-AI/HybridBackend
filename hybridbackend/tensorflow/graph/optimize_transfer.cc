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

#if HYBRIDBACKEND_TENSORFLOW

#include <map>
#include <set>
#include <vector>

#include <absl/strings/str_cat.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/public/session_options.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/graph/common/packing.h"
#include "hybridbackend/tensorflow/graph/common/replacing.h"
#include "hybridbackend/tensorflow/graph/common/rewriting.h"
#include "hybridbackend/tensorflow/graph/op_optimization.h"

namespace tensorflow {
namespace hybridbackend {

namespace {
inline bool CheckTransferOptimizationDisabled() {
  bool disabled =
      ::hybridbackend::EnvVarGetBool("HB_TRANSFER_OPTIMIZATION_DISABLED", true);
  if (disabled) {
    return true;
  }
  int gpu_count = 0;
  int gpu_major = 0;
  int gpu_minor = 0;
  ::hybridbackend::EnvGetGpuInfo(&gpu_count, &gpu_major, &gpu_minor);
  if (gpu_count < 1 || gpu_major < 8) {
    return true;
  }
  return false;
}

inline bool TransferOptimizationDisabled() {
  static const bool kTransferOptimizationDisabled =
      CheckTransferOptimizationDisabled();
  return kTransferOptimizationDisabled;
}

inline int TransferPackingBuckets() {
  static const int kTransferPackingBuckets =
      ::hybridbackend::EnvVarGetInt("HB_TRANSFER_PACKING_BUCKETS", 2);
  return kTransferPackingBuckets;
}
}  // namespace

class OptimizeTransferReductionPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    if (TF_PREDICT_FALSE(disabled)) {
      return Status::OK();
    }

    if (TF_PREDICT_TRUE(TransferOptimizationDisabled())) {
      return Status::OK();
    }

    // Scan H2D edges
    std::vector<const Edge*> h2d_edges;
    for (Node* node : graph->op_nodes()) {
      if (!node->IsOp()) {
        continue;
      }

      DeviceNameUtils::ParsedName assigned_device;
      if (!DeviceNameUtils::ParseFullName(node->assigned_device_name(),
                                          &assigned_device)) {
        continue;
      }

      if (assigned_device.type != DEVICE_CPU) {
        continue;
      }

      for (const auto& edge : node->out_edges()) {
        if (!edge) {
          continue;
        }
        if (edge->IsControlEdge()) {
          continue;
        }

        DeviceNameUtils::ParsedName out_assigned_device;
        if (!DeviceNameUtils::ParseFullName(edge->dst()->assigned_device_name(),
                                            &out_assigned_device)) {
          continue;
        }
        if (assigned_device.job != out_assigned_device.job ||
            assigned_device.replica != out_assigned_device.replica ||
            assigned_device.task != out_assigned_device.task) {
          continue;
        }
        if (out_assigned_device.type != DEVICE_GPU) {
          continue;
        }

        h2d_edges.emplace_back(edge);
      }
    }

    // Build H2D transfer ops
    std::map<const Node*, std::map<int, Node*>> identity_cache;
    std::vector<std::pair<Node*, int>> identity_srcs;
    std::vector<std::pair<Node*, int>> identity_dsts;
    for (auto h2d_edge : h2d_edges) {
      Node* src = h2d_edge->src();
      const int src_output = h2d_edge->src_output();
      auto& cache = identity_cache[src];
      auto it = cache.find(src_output);
      Node* identity_op = nullptr;
      if (it == cache.end()) {
        TF_RETURN_IF_ERROR(
            NodeBuilder(graph->NewName(
                            absl::StrCat(src->name(), "/output", src_output)),
                        "HbH2DTransfer")
                .Device(h2d_edge->dst()->assigned_device_name())
                .AssignedDevice(h2d_edge->dst()->assigned_device_name())
                .Input(NodeBuilder::NodeOut(src, src_output))
                .Finalize(graph, &identity_op));
        cache.emplace(src_output, identity_op);
      } else {
        identity_op = it->second;
      }
      identity_srcs.emplace_back(identity_op, 0);
      identity_dsts.emplace_back(h2d_edge->dst(), h2d_edge->dst_input());
    }

    // Insert H2D transfer ops
    for (size_t i = 0; i < identity_dsts.size(); ++i) {
      VLOG(2) << "Inserted transfer " << identity_srcs[i].first->name()
              << " before " << identity_dsts[i].first->name() << ":"
              << identity_dsts[i].second;
      graph->UpdateEdge(identity_srcs[i].first, identity_srcs[i].second,
                        identity_dsts[i].first, identity_dsts[i].second);
    }

    // Fuse H2D transfer ops
    const std::vector<DataType> transfer_dtypes{DT_FLOAT, DT_INT64, DT_UINT64};

    TF_RETURN_IF_ERROR(Pack("HbH2DTransfer", "HbH2DTransferN")
                           .WithDevice(DEVICE_GPU)
                           .WithTypeAttr("T", transfer_dtypes)
                           .WithBuckets(TransferPackingBuckets())
                           .In(graph));

    TF_RETURN_IF_ERROR(Replace("HbH2DTransfer", "Identity").In(graph));

    return Status::OK();
  }
};

REGISTER_REDUCTION_OPTIMIZATION(OptimizeTransferReductionPass);

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

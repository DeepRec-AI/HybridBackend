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

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/graph/common/packing.h"
#include "hybridbackend/tensorflow/graph/common/rewriting.h"
#include "hybridbackend/tensorflow/graph/op_optimization.h"

namespace tensorflow {
namespace hybridbackend {

namespace {
inline bool PrefetchPackingDisabled() {
  static const bool kPrefetchPackingDisabled =
      ::hybridbackend::EnvVarGetBool("HB_OP_PREFETCH_PACKING_DISABLED", true);
  return kPrefetchPackingDisabled;
}
}  // namespace

class OptimizePrefetchReplacingPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    TF_RETURN_IF_ERROR(Rewrite("CancelPrefetch", "HbCancelPrefetch")
                           .WithIntAttr("num_takers", 1)
                           .WithIntAttr("num_runners", 1)
                           .WithIntAttr("capacity", 1)
                           .WithIntAttr("memory_limit", 0)
                           .WithStrAttr("container", "")
                           .WithStrAttr("shared_name", "")
                           .In(graph));
    TF_RETURN_IF_ERROR(Rewrite("ResumePrefetch", "HbResumePrefetch")
                           .WithIntAttr("num_takers", 1)
                           .WithIntAttr("num_runners", 1)
                           .WithIntAttr("capacity", 1)
                           .WithIntAttr("memory_limit", 0)
                           .WithStrAttr("container", "")
                           .WithStrAttr("shared_name", "")
                           .In(graph));
    TF_RETURN_IF_ERROR(Rewrite("StopPrefetch", "HbStopPrefetch")
                           .WithIntAttr("num_takers", 1)
                           .WithIntAttr("num_runners", 1)
                           .WithIntAttr("capacity", 1)
                           .WithIntAttr("memory_limit", 0)
                           .WithStrAttr("container", "")
                           .WithStrAttr("shared_name", "")
                           .In(graph));
    TF_RETURN_IF_ERROR(Rewrite("RunPrefetch", "HbRunPrefetch")
                           .WithIntAttr("num_takers", 1)
                           .WithIntAttr("num_runners", 1)
                           .WithIntAttr("capacity", 1)
                           .WithIntAttr("memory_limit", 0)
                           .WithStrAttr("container", "")
                           .WithStrAttr("shared_name", "")
                           .In(graph));
    TF_RETURN_IF_ERROR(Rewrite("TakeFromPrefetch", "HbTakeFromPrefetch")
                           .WithIntAttr("num_takers", 1)
                           .WithIntAttr("num_runners", 1)
                           .WithIntAttr("capacity", 1)
                           .WithIntAttr("memory_limit", 0)
                           .WithStrAttr("container", "")
                           .WithStrAttr("shared_name", "")
                           .WithTypeListAttr("dtypes")
                           .In(graph));

    return Status::OK();
  }
};

REGISTER_REPLACING_OPTIMIZATION(OptimizePrefetchReplacingPass);

class OptimizePrefetchReductionPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    if (TF_PREDICT_FALSE(disabled)) {
      return Status::OK();
    }

    if (TF_PREDICT_TRUE(PrefetchPackingDisabled())) {
      return Status::OK();
    }

    // Scan run prefetch ops
    std::vector<Node*> run_prefetch_ops;
    for (Node* node : graph->op_nodes()) {
      if (!node->IsOp()) {
        continue;
      }

      if (node->type_string() == "HbRunPrefetch") {
        run_prefetch_ops.emplace_back(node);
      }
    }
    if (run_prefetch_ops.size() < 1) {
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
                        "HbH2DPrefetchedTransfer")
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
    TF_RETURN_IF_ERROR(
        Pack("HbH2DPrefetchedTransfer", "HbH2DPrefetchedTransferN")
            .WithDevice(DEVICE_GPU)
            .WithTypeAttr("T", {})
            .In(graph));

    return Status::OK();
  }
};

REGISTER_REDUCTION_OPTIMIZATION(OptimizePrefetchReductionPass);

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

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

#include <algorithm>
#include <deque>
#include <set>
#include <unordered_map>
#include <vector>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>

#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/core/util/device_name_utils.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/graph/common/helper.h"
#include "hybridbackend/tensorflow/graph/common/linearization.h"

namespace tensorflow {
namespace hybridbackend {
LinearizeOutputs::LinearizeOutputs(const string& op_type,
                                   const int32& op_output)
    : op_type_(op_type), op_output_(op_output) {}

Status LinearizeOutputs::In(Graph* graph) {
  std::unordered_map<Node*, int> candidates;
  std::vector<bool> dependencies;

  std::vector<Node*> sorted;
  GetReversePostOrder(*graph, &sorted, NodeComparatorName{});

  for (Node* node : graph->op_nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    if (node->type_string() != op_type_) {
      continue;
    }

    std::vector<Node*> linear_ops;
    for (Node* n : sorted) {
      for (const auto& edge : n->in_edges()) {
        if (edge && !edge->IsControlEdge() && edge->src() == node &&
            edge->src_output() == op_output_) {
          linear_ops.push_back(edge->dst());
          break;
        }
      }
    }

    if (linear_ops.size() < 2) {
      continue;
    }

    for (size_t idx = 1; idx < linear_ops.size(); ++idx) {
      graph->AddControlEdge(linear_ops[idx - 1], linear_ops[idx]);
    }

    VLOG(1) << "Linearized " << linear_ops.size() << " outputs of "
            << node->name() << " in graph " << static_cast<void*>(graph);
  }

  return Status::OK();
}

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

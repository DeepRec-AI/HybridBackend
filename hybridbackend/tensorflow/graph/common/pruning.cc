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

#include <set>
#include <string>
#include <vector>

#include <tensorflow/core/graph/node_builder.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/graph/common/pruning.h"

namespace tensorflow {
namespace hybridbackend {

Status InputPruneN(Graph* graph, const string& target_op_type,
                   const string& target_n_attr, const int& target_n_input,
                   const std::vector<string>& op_types,
                   const std::vector<int>& src_outputs,
                   const std::vector<int>& dst_inputs) {
  size_t search_depth = src_outputs.size();
  if (TF_PREDICT_FALSE(search_depth != dst_inputs.size())) {
    return errors::InvalidArgument(
        "src_outputs and dst_inputs must have same size");
  }

  // Scan for matched hyper edges
  std::vector<std::pair<Node*, int>> from_outputs;
  std::vector<std::pair<Node*, int>> to_outputs;
  std::vector<std::pair<Node*, int>> from_inputs;
  for (Node* node : graph->op_nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    if (node->type_string() != target_op_type) {
      continue;
    }

    int n = 1;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), target_n_attr, &n));

    for (int idx = n * target_n_input; idx < (n + 1) * target_n_input; ++idx) {
      const Edge* dst_edge;
      auto status = node->input_edge(idx, &dst_edge);
      if (!status.ok()) {
        continue;
      }
      if (dst_edge->IsControlEdge()) {
        continue;
      }

      Node* cursor = dst_edge->src();
      int cursor_src_output = dst_edge->src_output();
      bool matched = true;
      for (size_t d = 0; d < search_depth; ++d) {
        size_t ridx = search_depth - 1 - d;
        if (cursor->type_string() != op_types[ridx]) {
          matched = false;
          break;
        }
        if (cursor_src_output != src_outputs[ridx]) {
          matched = false;
          break;
        }

        const Edge* edge;
        status = cursor->input_edge(dst_inputs[ridx], &edge);
        if (!status.ok()) {
          matched = false;
          break;
        }
        if (edge->IsControlEdge()) {
          matched = false;
          break;
        }

        cursor = edge->src();
        cursor_src_output = edge->src_output();
      }
      if (!matched) {
        continue;
      }
      from_inputs.emplace_back(node, idx);
      from_outputs.emplace_back(dst_edge->src(), dst_edge->src_output());
      to_outputs.emplace_back(cursor, cursor_src_output);
    }
  }

  // Scan for dependencies to matched hyper edges
  const size_t num_matches = from_outputs.size();
  for (size_t i = 0; i < num_matches; ++i) {
    std::vector<const Edge*> edges;
    for (const auto& edge : from_outputs[i].first->out_edges()) {
      if (edge->IsControlEdge()) {
        continue;
      }
      if (edge->src_output() == from_outputs[i].second &&
          edge->dst() != from_inputs[i].first) {
        from_inputs.emplace_back(edge->dst(), edge->dst_input());
        from_outputs.emplace_back(from_outputs[i]);
        to_outputs.emplace_back(to_outputs[i]);
      }
    }
  }

  // Prune edges
  for (size_t i = 0; i < from_inputs.size(); ++i) {
    VLOG(1) << "Pruned input for " << from_inputs[i].first->name() << ": "
            << from_outputs[i].first->name() << ":" << from_outputs[i].second
            << " -> " << to_outputs[i].first->name() << ":"
            << to_outputs[i].second;
    graph->UpdateEdge(to_outputs[i].first, to_outputs[i].second,
                      from_inputs[i].first, from_inputs[i].second);
  }

  return Status::OK();
}

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

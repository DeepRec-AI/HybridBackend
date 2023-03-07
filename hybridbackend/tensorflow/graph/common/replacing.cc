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
#include <unordered_map>
#include <vector>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>

#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/util/device_name_utils.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/graph/common/helper.h"
#include "hybridbackend/tensorflow/graph/common/replacing.h"

namespace tensorflow {
namespace hybridbackend {

Replace::Replace(const string& op_type, const string& optimized_op_type)
    : op_type_(op_type),
      optimized_op_type_(optimized_op_type),
      device_(DEVICE_GPU),
      packed_(false) {}

Replace& Replace::WithDevice(const string& device) {
  device_ = device;
  return *this;
}

Replace& Replace::WithTypeAttr(const string& attr_name,
                               const std::vector<DataType>& constraints) {
  type_attrs_[attr_name] = constraints;
  return *this;
}

Replace& Replace::WithExtraIntAttr(const string& attr_name) {
  extra_int_attrs_.push_back(attr_name);
  return *this;
}

Replace& Replace::Packed() {
  packed_ = true;
  return *this;
}

Status Replace::In(Graph* graph) { return In(graph, nullptr); }

Status Replace::In(Graph* graph, int64* poccurrence_count) {
  int64 occurrence_count = 0;
  for (Node* node : graph->op_nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    if (node->type_string() != op_type_) {
      continue;
    }

    const int32 num_inputs = node->num_inputs();
    if (num_inputs < 1) {
      VLOG(2) << "Skipped replacing op " << node->name() << " since no inputs";
      continue;
    }

    DeviceNameUtils::ParsedName requested_device;
    if (!DeviceNameUtils::ParseFullName(node->requested_device(),
                                        &requested_device)) {
      VLOG(2) << "Skipped replacing op " << node->name()
              << " since no device set";
      continue;
    }

    if (requested_device.type != device_ || requested_device.job == "ps") {
      VLOG(2) << "Skipped replacing op " << node->name()
              << " since device differs, required /" << device_
              << ":0 but requested: " << node->requested_device();
      continue;
    }

    bool types_are_incompatible = false;
    for (auto& attr_pair : type_attrs_) {
      if (!HasNodeAttr(node->def(), attr_pair.first)) {
        types_are_incompatible = true;
        break;
      }
      DataType attrval;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attr_pair.first, &attrval));
      if (attr_pair.second.size() == 0) {
        continue;
      }
      types_are_incompatible = true;
      for (DataType dt : attr_pair.second) {
        if (dt == attrval) {
          types_are_incompatible = false;
          break;
        }
      }
    }

    if (TF_PREDICT_FALSE(types_are_incompatible)) {
      VLOG(2) << "Skipped replacing op " << node->name()
              << " since types incompatible";
      continue;
    }

    auto name = node->name();
    NodeBuilder replaced_op_builder = NodeBuilder(name, optimized_op_type_);
    replaced_op_builder.Device(node->requested_device());
    replaced_op_builder.AssignedDevice(node->assigned_device_name());
    for (const auto& iter : node->attrs()) {
      replaced_op_builder.Attr(iter.first, iter.second);
    }

    if (packed_) {
      replaced_op_builder.Attr("N", 1);
    }

    std::vector<Node*> control_inputs;
    for (const auto& edge : node->in_edges()) {
      if (!edge) {
        continue;
      }
      if (edge->IsControlEdge()) {
        control_inputs.push_back(edge->src());
      }
    }
    replaced_op_builder.ControlInputs(control_inputs);

    for (int32 idx = 0; idx < node->num_inputs(); ++idx) {
      const Edge* edge;
      TF_RETURN_IF_ERROR(node->input_edge(idx, &edge));
      if (packed_) {
        replaced_op_builder.Input(gtl::ArraySlice<NodeBuilder::NodeOut>{
            NodeBuilder::NodeOut(edge->src(), edge->src_output())});
      } else {
        replaced_op_builder.Input(
            NodeBuilder::NodeOut(edge->src(), edge->src_output()));
      }
    }

    Node* replaced_op_node = nullptr;
    TF_RETURN_IF_ERROR(replaced_op_builder.Finalize(graph, &replaced_op_node));
    VLOG(1) << "Replaced " << op_type_ << " op " << name << " on "
            << node->requested_device() << " in graph "
            << static_cast<void*>(graph);
    occurrence_count++;

    for (const auto& edge : node->out_edges()) {
      if (!edge) {
        continue;
      }
      if (edge->IsControlEdge()) {
        graph->AddControlEdge(replaced_op_node, edge->dst());
      } else {
        graph->AddEdge(replaced_op_node, edge->src_output(), edge->dst(),
                       edge->dst_input());
      }
    }

    graph->RemoveNode(node);
  }

  if (poccurrence_count != nullptr) {
    *poccurrence_count = occurrence_count;
  }
  return Status::OK();
}

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

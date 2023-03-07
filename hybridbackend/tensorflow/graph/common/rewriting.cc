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

#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/util/device_name_utils.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/graph/common/helper.h"
#include "hybridbackend/tensorflow/graph/common/rewriting.h"

namespace tensorflow {
namespace hybridbackend {

Rewrite::Rewrite(const string& op_like_name, const string& op_name)
    : op_like_name_(absl::StrCat("_HB_", op_like_name)),
      op_name_(op_name),
      device_(DEVICE_GPU) {}

Rewrite& Rewrite::WithDevice(const string& device) {
  device_ = device;
  return *this;
}

Rewrite& Rewrite::WithTypeAttr(const string& attr_name,
                               const DataType& default_attr) {
  type_attrs_[attr_name] = default_attr;
  return *this;
}
Rewrite& Rewrite::WithShapeAttr(const string& attr_name,
                                const TensorShape& default_attr) {
  shape_attrs_[attr_name] = default_attr;
  return *this;
}
Rewrite& Rewrite::WithIntAttr(const string& attr_name,
                              const int32& default_attr) {
  int_attrs_[attr_name] = default_attr;
  return *this;
}
Rewrite& Rewrite::WithStrAttr(const string& attr_name,
                              const string& default_attr) {
  str_attrs_[attr_name] = default_attr;
  return *this;
}

Rewrite& Rewrite::WithTypeListAttr(const string& attr_name) {
  type_list_attrs_.push_back(attr_name);
  return *this;
}

Status Rewrite::In(Graph* graph) { return In(graph, nullptr); }

Status Rewrite::In(Graph* graph, int64* poccurrence_count) {
  int64 occurrence_count = 0;
  for (Node* node : graph->op_nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    // Output op should have op_like_name_ attribute.
    if (!HasNodeAttr(node->def(), op_like_name_)) {
      continue;
    }

    // Construct rewrited op.
    NodeBuilder rewrited_op_builder = NodeBuilder(node->name(), op_name_);

    // Place op to specific device.
    if (!device_.empty()) {
      DeviceNameUtils::ParsedName requested_device;
      if (!DeviceNameUtils::ParseFullName(node->requested_device(),
                                          &requested_device)) {
        continue;
      }
      requested_device.type = device_;
      rewrited_op_builder.Device(
          DeviceNameUtils::ParsedNameToString(requested_device));
    }

    // String attributes.
    for (auto& attr_pair : str_attrs_) {
      const string attrkey = absl::StrCat(op_like_name_, "_", attr_pair.first);
      string attrval = attr_pair.second;
      if (HasNodeAttr(node->def(), attrkey)) {
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attrkey, &attrval));
      }
      rewrited_op_builder.Attr(attr_pair.first, attrval);
    }

    // Integer attributes.
    for (auto& attr_pair : int_attrs_) {
      const string attrkey = absl::StrCat(op_like_name_, "_", attr_pair.first);
      int32 attrval = attr_pair.second;
      if (HasNodeAttr(node->def(), attrkey)) {
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attrkey, &attrval));
      }
      rewrited_op_builder.Attr(attr_pair.first, attrval);
    }

    // Data type attributes.
    for (auto& attr_pair : type_attrs_) {
      const string attrkey = absl::StrCat(op_like_name_, "_", attr_pair.first);
      DataType attrval = attr_pair.second;
      if (HasNodeAttr(node->def(), attrkey)) {
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attrkey, &attrval));
      }
      rewrited_op_builder.Attr(attr_pair.first, attrval);
    }

    // Tensor shape attributes.
    for (auto& attr_pair : shape_attrs_) {
      const string attrkey = absl::StrCat(op_like_name_, "_", attr_pair.first);
      TensorShape attrval = attr_pair.second;
      if (HasNodeAttr(node->def(), attrkey)) {
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attrkey, &attrval));
      }
      rewrited_op_builder.Attr(attr_pair.first, attrval);
    }

    // Data type list attributes.
    for (auto& attr : type_list_attrs_) {
      const string attrkey = absl::StrCat(op_like_name_, "_", attr);
      std::vector<DataType> attrval;
      if (HasNodeAttr(node->def(), attrkey)) {
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attrkey, &attrval));
      }
      rewrited_op_builder.Attr(attr, attrval);
    }

    std::vector<Node*> control_inputs;
    const string input_proxy_key = absl::StrCat(op_like_name_, "_input_proxy");
    std::unordered_map<Node*, int> input_proxies;
    for (const auto& edge : node->in_edges()) {
      if (!edge) {
        continue;
      }
      if (!edge->IsControlEdge()) {
        continue;
      }
      Node* control_input = edge->src();
      if (HasNodeAttr(control_input->def(), input_proxy_key)) {
        int32 input_idx = -1;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(control_input->attrs(), input_proxy_key, &input_idx));
        if (input_idx < 0) {
          LOG(ERROR) << "Invalid attribute set for input proxy: "
                     << control_input->name();
          control_inputs.push_back(control_input);
        } else {
          input_proxies[control_input] = input_idx;
        }
      } else {
        control_inputs.push_back(control_input);
      }
    }

    std::vector<Node*> input_proxy_vec(input_proxies.size(), nullptr);
    for (auto& it : input_proxies) {
      if (it.second < 0 || it.second > input_proxy_vec.size() - 1) {
        return errors::Internal("Invalid attribute set for input proxy: ",
                                it.first->name());
      }
      input_proxy_vec[it.second] = it.first;
    }
    for (size_t idx = 0; idx < input_proxy_vec.size(); ++idx) {
      if (input_proxy_vec[idx] == nullptr) {
        return errors::Internal("Invalid attribute set for input proxy ", idx);
      }
    }

    for (Node* input_proxy : input_proxy_vec) {
      if (input_proxy->type_string() == "Identity") {
        const Edge* edge;
        TF_RETURN_IF_ERROR(input_proxy->input_edge(0, &edge));
        rewrited_op_builder.Input(
            NodeBuilder::NodeOut(edge->src(), edge->src_output()));
      } else if (input_proxy->type_string() == "IdentityN") {
        std::vector<NodeBuilder::NodeOut> list_inputs;
        list_inputs.reserve(input_proxy->num_inputs());
        for (int32 idx = 0; idx < input_proxy->num_inputs(); ++idx) {
          const Edge* edge;
          TF_RETURN_IF_ERROR(input_proxy->input_edge(idx, &edge));
          list_inputs.emplace_back(edge->src(), edge->src_output());
        }
        rewrited_op_builder.Input(list_inputs);
      } else {
        return errors::Internal("Invalid type ", input_proxy->type_string(),
                                " for input proxy ", input_proxy->name());
      }
    }
    rewrited_op_builder.ControlInputs(control_inputs);

    Node* rewrited_op_node = nullptr;
    TF_RETURN_IF_ERROR(rewrited_op_builder.Finalize(graph, &rewrited_op_node));
    occurrence_count++;
    VLOG(1) << "Rewrited function " << op_like_name_ << " to " << op_name_
            << " op " << rewrited_op_node->name() << " in graph "
            << static_cast<void*>(graph);

    for (const auto& edge : node->out_edges()) {
      if (!edge) {
        continue;
      }
      if (edge->IsControlEdge()) {
        graph->AddControlEdge(rewrited_op_node, edge->dst());
      } else {
        graph->AddEdge(rewrited_op_node, edge->src_output(), edge->dst(),
                       edge->dst_input());
      }
    }

    graph->RemoveNode(node);
    for (auto& it : input_proxies) {
      graph->RemoveNode(it.first);
    }
  }

  if (poccurrence_count != nullptr) {
    *poccurrence_count = occurrence_count;
  }
  return Status::OK();
}

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

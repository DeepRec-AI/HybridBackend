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
#include <sstream>
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
#include "hybridbackend/tensorflow/graph/common/packing.h"

namespace tensorflow {
namespace hybridbackend {

namespace {
inline bool PackingDisabled() {
  static const bool kPackingDisabled =
      ::hybridbackend::EnvVarGetBool("HB_OP_PACKING_DISABLED", false);
  return kPackingDisabled;
}

inline string ClustersDebugString(
    const std::unordered_map<string, std::vector<Node*>>& clusters) {
  std::ostringstream debug_sstr;
  size_t cluster_size = clusters.size();
  size_t idx = 0;
  for (auto& c : clusters) {
    debug_sstr << c.first << " (" << c.second.size() << ")";
    if (idx < cluster_size - 1) {
      debug_sstr << ", ";
    }
    ++idx;
  }
  return debug_sstr.str();
}
}  // namespace

Pack::Pack(const string& op_type, const string& optimized_op_type)
    : op_type_(op_type),
      optimized_op_type_(optimized_op_type),
      device_(DEVICE_GPU) {
  static const int kDefaultPackingNBuckets =
      ::hybridbackend::EnvVarGetInt("HB_OP_PACKING_NBUCKETS", 1);
  num_buckets_ = kDefaultPackingNBuckets;
}

Pack& Pack::WithDevice(const string& device) {
  device_ = device;
  return *this;
}

Pack& Pack::WithTypeAttr(const string& attr_name,
                         const std::vector<DataType>& constraints) {
  type_attrs_[attr_name] = constraints;
  return *this;
}

Pack& Pack::WithShapeAttr(const string& attr_name) {
  shape_attrs_.push_back(attr_name);
  return *this;
}

Pack& Pack::WithIntAttr(const string& attr_name) {
  int_attrs_.push_back(attr_name);
  return *this;
}

Pack& Pack::WithStrAttr(const string& attr_name) {
  str_attrs_.push_back(attr_name);
  return *this;
}

Pack& Pack::WithAggregatedShapeAttr(const string& attr_name) {
  aggregated_shape_attrs_.push_back(attr_name);
  return *this;
}

Pack& Pack::WithAggregatedIntAttr(const string& attr_name) {
  aggregated_int_attrs_.push_back(attr_name);
  return *this;
}

Pack& Pack::WithAggregatedStrAttr(const string& attr_name) {
  aggregated_str_attrs_.push_back(attr_name);
  return *this;
}

Pack& Pack::WithHandle(const int32 input) {
  handles_.push_back(input);
  return *this;
}

Pack& Pack::WithBuckets(const int32 num_buckets) {
  num_buckets_ = num_buckets;
  return *this;
}

Status Pack::In(Graph* graph) { return In(graph, nullptr); }

Status Pack::In(Graph* graph, int64* poccurrence_count) {
  int64 occurrence_count = 0;
  std::unordered_map<Node*, int> candidates;
  std::vector<bool> dependencies;

  std::vector<Node*> sorted;
  GetReversePostOrder(*graph, &sorted);

  std::vector<Node*> roots;
  int seq = 0;
  for (Node* node : sorted) {
    if (!node->IsOp()) {
      continue;
    }

    if (node->type_string() != op_type_) {
      continue;
    }

    const int32 num_inputs = node->num_inputs();
    if (num_inputs < 1) {
      VLOG(2) << "Skipped packing of op " << node->name() << " since no inputs";
      continue;
    }

    DeviceNameUtils::ParsedName requested_device;
    if (!DeviceNameUtils::ParseFullName(node->requested_device(),
                                        &requested_device)) {
      VLOG(2) << "Skipped packing of op " << node->name()
              << " since no device set";
      continue;
    }

    if (requested_device.type != device_ || requested_device.job == "ps") {
      VLOG(2) << "Skipped packing of op " << node->name()
              << " since device differs: /" << device_ << ":0 vs. "
              << node->requested_device();
      continue;
    }

    bool types_are_incompatible = false;
    for (auto& attr_pair : type_attrs_) {
      if (!HasNodeAttr(node->def(), attr_pair.first)) {
        VLOG(2) << "Skipped packing of op " << node->name() << " since "
                << op_type_ << " nodes has no attribute " << attr_pair.first;
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
      if (TF_PREDICT_FALSE(types_are_incompatible)) {
        VLOG(2) << "Skipped packing of op " << node->name() << " since "
                << op_type_ << " nodes with attribute " << attr_pair.first
                << "=" << DataTypeString(attrval) << " is not supported";
      }
    }

    if (TF_PREDICT_FALSE(types_are_incompatible)) {
      continue;
    }

    roots.push_back(node);
    candidates.emplace(node, seq++);
  }
  const size_t size = candidates.size();
  if (size == 0) {
    return Status::OK();
  }

  dependencies.resize(size * size, false);
  for (size_t i = 0; i < size; ++i) {
    dependencies[i * size + i] = true;
  }

  for (size_t i = 0; i < size; ++i) {
    Node* root = roots[i];
    auto root_it = candidates.find(root);
    DFSFrom(*graph, {root},
            [&](Node* n) {
              auto it = candidates.find(n);
              if (it != candidates.end()) {
                dependencies[root_it->second * size + it->second] = true;
              }
            },
            nullptr);
  }

  if (TF_PREDICT_FALSE(PackingDisabled())) {
    return Status::OK();
  }

  std::vector<int> colors(size, 0);
  for (int color = 0; color < size; ++color) {
    int cursor = 0;
    bool color_is_set = false;
    for (; cursor < size; ++cursor) {
      if (colors[cursor] != color) {
        continue;
      }
      for (size_t i = cursor + 1; i < size; ++i) {
        if (dependencies[cursor * size + i]) {
          colors[i] = color + 1;
        }
      }
      color_is_set = true;
    }
    if (!color_is_set) {
      break;
    }
  }

  std::unordered_map<string, std::vector<Node*>> scoped_clusters;
  for (auto candidate : candidates) {
    Node* node = candidate.first;

    std::vector<string> attrs;
    for (auto& attr_pair : type_attrs_) {
      if (HasNodeAttr(node->def(), attr_pair.first)) {
        DataType attrval;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(node->attrs(), attr_pair.first, &attrval));
        attrs.push_back(DataTypeString(attrval));
      } else {
        attrs.push_back("_");
      }
    }
    for (auto& attr : shape_attrs_) {
      if (HasNodeAttr(node->def(), attr)) {
        TensorShape attrval;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attr, &attrval));
        attrs.push_back(attrval.DebugString());
      } else {
        attrs.push_back("_");
      }
    }
    for (auto& attr : int_attrs_) {
      if (HasNodeAttr(node->def(), attr)) {
        int attrval;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attr, &attrval));
        attrs.push_back(std::to_string(attrval));
      } else {
        attrs.push_back("_");
      }
    }
    for (auto& attr : str_attrs_) {
      if (HasNodeAttr(node->def(), attr)) {
        string attrval;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), attr, &attrval));
        attrs.push_back(attrval);
      } else {
        attrs.push_back("_");
      }
    }

    std::vector<string> handles;
    for (auto& idx : handles_) {
      const Edge* input_edge;
      TF_RETURN_IF_ERROR(node->input_edge(idx, &input_edge));
      handles.push_back(input_edge->src()->name());
    }

    auto name = node->name();
    auto prefix = name.substr(0, name.find_first_of('/'));

    int seq = candidates[node];
    string cluster_key =
        absl::StrCat(colors[seq], ":", absl::StrJoin(attrs, "|"), ":",
                     absl::StrJoin(handles, "|"), ":", prefix);
    auto it = scoped_clusters.find(cluster_key);
    if (it == scoped_clusters.end()) {
      scoped_clusters.emplace(cluster_key, std::vector<Node*>{node});
    } else {
      it->second.push_back(node);
    }
  }

  std::unordered_map<string, std::vector<Node*>> clusters;
  for (auto& c : scoped_clusters) {
    int cluster_size = c.second.size();
    if (cluster_size < 2) {
      continue;
    }
    if (cluster_size < num_buckets_ * 2) {
      clusters.emplace(c.first, c.second);
      continue;
    }
    std::vector<std::vector<Node*>> buckets(num_buckets_);
    for (size_t i = 0; i < cluster_size; ++i) {
      buckets[i % num_buckets_].push_back(c.second[i]);
    }
    for (size_t b = 0; b < num_buckets_; ++b) {
      clusters.emplace(absl::StrCat(c.first, ":", b), buckets[b]);
    }
  }

  VLOG(3) << "Try pack " << size << " " << op_type_ << " ops into "
          << clusters.size() << " clusters: " << ClustersDebugString(clusters);

  for (auto& c : clusters) {
    if (c.second.size() <= 1) {
      continue;
    }

    Node* node0 = c.second[0];
    std::unordered_map<string, AttrValue> node0attrs;
    for (const auto& iter : node0->attrs()) {
      node0attrs[iter.first] = iter.second;
    }
    bool all_attrs_are_equal = true;
    for (size_t i = 1; i < c.second.size(); ++i) {
      if (node0->num_inputs() != c.second[i]->num_inputs()) {
        VLOG(2) << "Skipped packing of " << op_type_ << " ops (" << c.first
                << ") with " << c.second.size()
                << " nodes since num_inputs differs: " << node0->num_inputs()
                << " vs. " << c.second[i]->num_inputs();
        all_attrs_are_equal = false;
        break;
      }
      if (node0->num_outputs() != c.second[i]->num_outputs()) {
        VLOG(2) << "Skipped packing of " << op_type_ << " ops (" << c.first
                << ") with " << c.second.size()
                << " nodes since num_outputs differs: " << node0->num_outputs()
                << " vs. " << c.second[i]->num_outputs();
        all_attrs_are_equal = false;
        break;
      }
      if (node0->requested_device() != c.second[i]->requested_device()) {
        all_attrs_are_equal = false;
        VLOG(2) << "Skipped packing of " << op_type_ << " ops (" << c.first
                << ") with " << c.second.size()
                << " nodes since requested device differs: "
                << node0->requested_device() << " vs. "
                << c.second[i]->requested_device();
        break;
      }
      for (const auto& iter : c.second[i]->attrs()) {
        if (iter.first == "_class") {
          continue;
        }
        if (std::find(aggregated_shape_attrs_.begin(),
                      aggregated_shape_attrs_.end(),
                      iter.first) != aggregated_shape_attrs_.end()) {
          continue;
        }
        if (std::find(aggregated_int_attrs_.begin(),
                      aggregated_int_attrs_.end(),
                      iter.first) != aggregated_int_attrs_.end()) {
          continue;
        }
        if (std::find(aggregated_str_attrs_.begin(),
                      aggregated_str_attrs_.end(),
                      iter.first) != aggregated_str_attrs_.end()) {
          continue;
        }
        auto iter0 = node0attrs.find(iter.first);
        if (iter0 == node0attrs.end()) {
          all_attrs_are_equal = false;
          VLOG(2) << "Skipped packing of " << op_type_ << " ops (" << c.first
                  << ") with " << c.second.size()
                  << " nodes since attribute not found: " << iter.first;
          break;
        }
        if (!AreAttrValuesEqual(iter.second, iter0->second)) {
          all_attrs_are_equal = false;
          VLOG(2) << "Skipped packing of " << op_type_ << " ops (" << c.first
                  << ") with " << c.second.size() << " nodes since attribute "
                  << iter.first << " differs";
          break;
        }
      }
      if (!all_attrs_are_equal) {
        break;
      }
    }
    if (!all_attrs_are_equal) {
      continue;
    }

    std::sort(c.second.begin(), c.second.end(), NodeComparatorName{});

    std::vector<std::vector<string>> fragments;
    for (size_t n = 0; n < c.second.size(); ++n) {
      std::vector<string> splits = absl::StrSplit(c.second[n]->name(), '/');
      for (size_t i = 0; i < splits.size(); ++i) {
        if (i < fragments.size()) {
          fragments[i].push_back(splits[i]);
        } else {
          std::vector<string> new_splits{splits[i]};
          fragments.emplace_back(std::move(new_splits));
        }
      }
    }

    std::vector<string> new_node_fragments;
    for (auto& s : fragments) {
      std::sort(s.begin(), s.end());
      auto last = std::unique(s.begin(), s.end());
      s.erase(last, s.end());
      if (s.size() < 2) {
        new_node_fragments.push_back(s[0]);
        continue;
      }
      new_node_fragments.push_back(
          absl::StrCat(s[0], "__", s.size(), "_packed"));
    }
    const string new_node_name =
        graph->NewName(absl::StrJoin(new_node_fragments, "/"));
    NodeBuilder packed_op_builder =
        NodeBuilder(new_node_name, optimized_op_type_);
    packed_op_builder.Device(node0->requested_device());
    packed_op_builder.AssignedDevice(node0->assigned_device_name());
    for (const auto& iter : node0->attrs()) {
      if (iter.first == "N") {
        continue;
      }
      if (std::find(aggregated_shape_attrs_.begin(),
                    aggregated_shape_attrs_.end(),
                    iter.first) != aggregated_shape_attrs_.end()) {
        continue;
      }
      if (std::find(aggregated_int_attrs_.begin(), aggregated_int_attrs_.end(),
                    iter.first) != aggregated_int_attrs_.end()) {
        continue;
      }
      if (std::find(aggregated_str_attrs_.begin(), aggregated_str_attrs_.end(),
                    iter.first) != aggregated_str_attrs_.end()) {
        continue;
      }
      packed_op_builder.Attr(iter.first, iter.second);
    }
    packed_op_builder.Attr("N", static_cast<int>(c.second.size()));
    for (const auto& attr_name : aggregated_shape_attrs_) {
      std::vector<TensorShape> attr_values;
      for (Node* n : c.second) {
        TensorShape attrval;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), attr_name, &attrval));
        attr_values.push_back(attrval);
      }
      packed_op_builder.Attr(attr_name, attr_values);
    }
    for (const auto& attr_name : aggregated_int_attrs_) {
      std::vector<int32> attr_values;
      for (Node* n : c.second) {
        int32 attrval;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), attr_name, &attrval));
        attr_values.push_back(attrval);
      }
      packed_op_builder.Attr(attr_name, attr_values);
    }
    for (const auto& attr_name : aggregated_str_attrs_) {
      std::vector<string> attr_values;
      for (Node* n : c.second) {
        string attrval;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), attr_name, &attrval));
        attr_values.push_back(attrval);
      }
      packed_op_builder.Attr(attr_name, attr_values);
    }

    std::vector<Node*> control_inputs;
    for (Node* node : c.second) {
      for (const auto& edge : node->in_edges()) {
        if (!edge) {
          continue;
        }
        if (edge->IsControlEdge()) {
          control_inputs.push_back(edge->src());
        }
      }
    }
    packed_op_builder.ControlInputs(control_inputs);

    std::vector<bool> handles_mask(node0->num_inputs(), false);
    for (auto& idx : handles_) {
      handles_mask[idx] = true;
    }
    for (int32 idx = 0; idx < node0->num_inputs(); ++idx) {
      if (handles_mask[idx]) {
        const Edge* edge0;
        TF_RETURN_IF_ERROR(node0->input_edge(idx, &edge0));
        packed_op_builder.Input(edge0->src());
      } else {
        std::vector<NodeBuilder::NodeOut> input_slices;
        for (Node* node : c.second) {
          const Edge* edge;
          TF_RETURN_IF_ERROR(node->input_edge(idx, &edge));
          input_slices.emplace_back(edge->src(), edge->src_output());
        }
        packed_op_builder.Input(input_slices);
      }
    }

    Node* packed_op_node = nullptr;
    TF_RETURN_IF_ERROR(packed_op_builder.Finalize(graph, &packed_op_node));
    VLOG(1) << "Packed " << op_type_ << " ops (" << c.first << ") with "
            << c.second.size() << " nodes on "
            << packed_op_node->assigned_device_name() << ": "
            << NodeJoin(c.second, ", ") << " -> " << packed_op_node->name()
            << " in graph " << static_cast<void*>(graph);
    occurrence_count++;

    for (size_t i = 0; i < c.second.size(); ++i) {
      std::vector<NodeBuilder::NodeOut> outputs;
      outputs.reserve(c.second[i]->num_outputs());
      for (int32 o = 0; o < c.second[i]->num_outputs(); ++o) {
        outputs.emplace_back(packed_op_node, c.second.size() * o + i);
      }
      Node* output_op_node = nullptr;
      TF_RETURN_IF_ERROR(
          NodeBuilder(c.second[i]->name(), "IdentityN")
              .Device(packed_op_node->requested_device())
              .AssignedDevice(packed_op_node->assigned_device_name())
              .Input(outputs)
              .Finalize(graph, &output_op_node));

      std::vector<const Edge*> edges;
      for (const auto& edge : c.second[i]->out_edges()) {
        edges.push_back(edge);
      }
      for (const auto& edge : edges) {
        if (!edge) {
          continue;
        }
        if (edge->IsControlEdge()) {
          graph->AddControlEdge(output_op_node, edge->dst());
        } else {
          graph->AddEdge(output_op_node, edge->src_output(), edge->dst(),
                         edge->dst_input());
        }
        graph->RemoveEdge(edge);
      }

      candidates.erase(c.second[i]);
      graph->RemoveNode(c.second[i]);
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

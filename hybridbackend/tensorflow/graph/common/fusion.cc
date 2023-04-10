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

#include <absl/strings/str_cat.h>

#include "hybridbackend/tensorflow/graph/common/fusion.h"

namespace {
// helper to check whether a template node is an input or output node
inline bool ScanUnsignInteger(const char** str) {
  const char* begin = *str;
  while (**str != '\0' && **str <= '9' && **str >= '0') {
    ++(*str);
  }
  return *str > begin;
}

inline bool ScanInteger(const char** str) {
  if (**str == '+' || **str == '-') {
    ++(*str);
  }
  return ScanUnsignInteger(str);
}

inline bool IsAllNum(const char* str) {
  if (str == nullptr) return false;

  bool numeric = ScanInteger(&str);
  if (*str == '.') {
    ++str;
    numeric = ScanUnsignInteger(&str) || numeric;
  }
  if (*str == 'e' || *str == 'E') {
    ++str;
    numeric = numeric && ScanInteger(&str);
  }
  return numeric && *str == '\0';
}

inline void sortInEdges(
    const tensorflow::EdgeSet& in,
    std::vector<std::tuple<int, const tensorflow::Edge*>>& out) {
  for (auto* e : in) {
    out.emplace_back(std::make_tuple(e->dst_input(), e));
  }
  std::sort(out.begin(), out.end());
}
}  // namespace

namespace tensorflow {
namespace hybridbackend {

// a bundle of output edges attached to the
// same port and utilized by CheckDynamicInputsImpl and
// CheckDynamicOutputsImpl
class OutEdges {
 public:
  OutEdges() : remainEdgeVal(0) {}

  // add a new edge
  void Append(const Edge* item) {
    oedges.push_back(item);
    remainEdgeVal++;
  }

  // mapping a graph node_port to a tempalte node port
  // node_port: a graph node port (outputs)
  // node_max_port: the maximum port of the node (graph)
  // start_port: the template port index where the dynamic output edge starts
  // end_port: the template port index where the dynamic output edge ends
  // return: the port index on template
  int GetTemplatePort(const int node_port, const int node_max_port,
                      const int start_port, const int end_port);
  // if a template node has an existing matching, checking endpoints of
  // all edges; if edge endpoint matched, decreases the remained edge number
  // by one. Otherwise, if dy_mode is 1 (parallel), add a new node to both of
  // temp_node_map and matched_node_map
  // return true if any of its output edge has a node matched
  bool CheckMatchedNode(
      const fusion_template::NodeDesc& temp_output_node,
      fusion_template::NodeMatching& matched,
      std::map<const std::string, fusion_template::NodeDesc>& temp_node_map,
      std::map<std::string, fusion_template::NodeMatching>& matched_node_map,
      std::map<std::string, std::string>& node_to_temp_key, const int dy_mode);

  // if a template node is not in matched_node_map, trying to emplace it
  // return true if any of its output edge has a node being added
  bool AddNewNode(
      const fusion_template::NodeDesc& temp_output_node,
      const std::string& output_key,
      std::map<const std::string, fusion_template::NodeDesc>& temp_node_map,
      std::map<std::string, fusion_template::NodeMatching>& matched_node_map,
      std::map<std::string, std::string>& node_to_temp_key);

  int Size() { return oedges.size(); }
  int RemainEdge() { return remainEdgeVal; }
  std::vector<const Edge*> Get() { return oedges; }

 private:
  std::vector<const Edge*> oedges;
  int remainEdgeVal;
};

int OutEdges::GetTemplatePort(const int node_port, const int node_max_port,
                              const int start_port, const int end_port) {
  int output_port = -1;
  if (node_port >= start_port && node_port <= (node_max_port + end_port)) {
    output_port = start_port;
  } else {
    int dynamic_node_num = node_max_port + end_port - start_port + 1;
    if (dynamic_node_num >= 1) {
      dynamic_node_num--;
    }
    output_port =
        (node_port < start_port) ? node_port : (node_port - dynamic_node_num);
  }
  return output_port;
}

bool OutEdges::CheckMatchedNode(
    const fusion_template::NodeDesc& temp_output_node,
    fusion_template::NodeMatching& matched,
    std::map<const std::string, fusion_template::NodeDesc>& temp_node_map,
    std::map<std::string, fusion_template::NodeMatching>& matched_node_map,
    std::map<std::string, std::string>& node_to_temp_key, const int dy_mode) {
  bool ret = false;
  for (auto oedge_it = oedges.begin(); oedge_it != oedges.end(); ++oedge_it) {
    const Edge* oedge = *oedge_it;
    const Node* output_node = oedge->dst();
    if (output_node == matched.node || output_node->type_string() == "ShapeN") {
      ret = true;
      remainEdgeVal--;
    } else if (dy_mode == 1) {
      std::string expand_key =
          temp_output_node.key + "_" + std::to_string(matched.dy_offset);
      temp_node_map.emplace(expand_key, temp_output_node);
      fusion_template::NodeMatching dy_input_node(output_node);
      matched_node_map.emplace(expand_key, dy_input_node);
      node_to_temp_key.emplace(output_node->name(), expand_key);
      matched.dy_offset++;
      ret = true;
      remainEdgeVal--;
    }
  }
  return ret;
}

bool OutEdges::AddNewNode(
    const fusion_template::NodeDesc& temp_output_node,
    const std::string& output_key,
    std::map<const std::string, fusion_template::NodeDesc>& temp_node_map,
    std::map<std::string, fusion_template::NodeMatching>& matched_node_map,
    std::map<std::string, std::string>& node_to_temp_key) {
  bool ret = false;
  int64 template_id = 1;
  for (auto oedge_it = oedges.begin(); oedge_it != oedges.end(); ++oedge_it) {
    const Edge* oedge = *oedge_it;
    const Node* output_node = oedge->dst();
    if (output_node->type_string() == temp_output_node.op) {
      if (ret) {
        std::string expand_key = output_key + "_" + std::to_string(template_id);
        // add a new template node
        temp_node_map.emplace(expand_key, temp_output_node);
        // add a new matched node
        fusion_template::NodeMatching matched_node(output_node);
        matched_node_map.emplace(expand_key, matched_node);
        node_to_temp_key.emplace(output_node->name(), expand_key);
        template_id++;
      } else {
        fusion_template::NodeMatching matched_node(output_node);
        matched_node_map.emplace(temp_output_node.key, matched_node);
        node_to_temp_key.emplace(output_node->name(), temp_output_node.key);
        ret = true;
      }
      remainEdgeVal--;
    } else {
      if (output_node->type_string() == "ShapeN") {
        remainEdgeVal--;
      }
    }
  }
  return ret;
}

bool FusionTemplate::CheckDynamicInputsImpl(
    const Node* node, const fusion_template::NodeDesc* temp_node,
    const int dy_mode, std::vector<const Edge*>& fused_op_inputs,
    std::map<const std::string, fusion_template::NodeDesc>& temp_node_map,
    std::map<std::string, fusion_template::NodeMatching>& matched_node_map,
    const int32 start_port, const int32 end_port) {
  // find the maximal input port
  int max_port = 0;
  for (auto* iedge : node->in_edges()) {
    max_port = (iedge->dst_input() > max_port) ? iedge->dst_input() : max_port;
  }
  for (auto* iedge : node->in_edges()) {
    int input_port = iedge->dst_input();
    if (input_port < 0) {
      // filter out NoOp edges
      continue;
    }
    const Node* input_node = iedge->src();
    // find the src node key in template
    std::string temp_input_key = "";
    auto iter = node_to_temp_key_.find(input_node->name());
    if (iter != node_to_temp_key_.end()) {
      temp_input_key = iter->second;
    } else {
      if (dy_mode == 2) {
        // switch off dynamic checking
        temp_input_key = temp_node->inputs[input_port];
      } else if (input_port >= start_port &&
                 input_port < (max_port + 1 + end_port)) {
        // input_port is within the dynamic checking range
        temp_input_key = (input_port == start_port)
                             ? temp_node->inputs[start_port]
                             : (temp_node->inputs[start_port] + "_" +
                                std::to_string(input_port));
        if (input_port != start_port) {
          temp_node_map.emplace(temp_input_key,
                                temp_node_map[temp_node->inputs[start_port]]);
        }
      } else {
        // fallback to standard input edge/node
        int dynamic_node_num = max_port + 1 + end_port - start_port;
        if (dynamic_node_num >= 1) {
          dynamic_node_num--;
        }
        temp_input_key = (input_port < start_port)
                             ? temp_node->inputs[input_port]
                             : temp_node->inputs[input_port - dynamic_node_num];
      }
    }

    // from temp_input_key to temp node
    if (::IsAllNum(temp_input_key.c_str())) {
      fused_op_inputs[atoi(temp_input_key.c_str())] = iedge;
      continue;
    }

    // added for dynamic input edges
    if (temp_input_key == "*") {
      // add to dynamic input edges
      fused_op_inputs.push_back(iedge);
      continue;
    }

    const fusion_template::NodeDesc temp_input_node =
        temp_node_map[temp_input_key];
    if (input_node->type_string() == temp_input_node.op) {
      auto it = matched_node_map.find(temp_input_key);
      if (it != matched_node_map.end()) {
        // double check the returned node of matched_node_map
        // with the input_node
        if (input_node != it->second.node) {
          if (dy_mode == 1) {
            // add a new node to both of temp_node_map
            // and matched_node_map
            std::string expand_key = temp_input_node.key + "_" +
                                     std::to_string(it->second.dy_offset);
            temp_node_map.emplace(expand_key, temp_input_node);
            fusion_template::NodeMatching dy_input_node(input_node);
            matched_node_map.emplace(expand_key, dy_input_node);
            node_to_temp_key_.emplace(input_node->name(), expand_key);
            it->second.dy_offset++;
          } else {
            return false;
          }
        }
      } else {
        // add new item to matched_node_map
        fusion_template::NodeMatching matched_node(input_node);
        matched_node_map.emplace(temp_input_key, matched_node);
        node_to_temp_key_.emplace(input_node->name(), temp_input_key);
      }
      continue;
    }
    return false;
  }
  return true;
}

bool FusionTemplate::CheckDynamicOutputsImpl(
    const Node* node, const fusion_template::NodeDesc* temp_node,
    const int dy_mode, std::vector<std::vector<const Edge*>>& fused_op_outputs,
    std::map<const std::string, fusion_template::NodeDesc>& temp_node_map,
    std::map<std::string, fusion_template::NodeMatching>& matched_node_map,
    const int32 start_port, const int32 end_port) {
  // port to OutEdges
  std::map<int, OutEdges> oedge_map;
  int max_port = 0;
  // find the maximum port index
  for (auto* oedge : node->out_edges()) {
    int output_port = oedge->src_output();
    max_port = (output_port > max_port) ? output_port : max_port;
    if (output_port >= 0) {
      oedge_map[output_port].Append(oedge);
    }
  }
  // looping over all ports (one port to one tensor)
  for (auto iter = oedge_map.begin(); iter != oedge_map.end(); ++iter) {
    auto oedges = iter->second;
    // obtain the corresponding output port on template
    int output_port = (dy_mode == 2)
                          ? iter->first
                          : oedges.GetTemplatePort(iter->first, max_port,
                                                   start_port, end_port);
    if (output_port < 0) {
      return false;
    }

    const auto& output_keys = temp_node->outputs[output_port];
    bool outgoing_port = false;

    for (auto& output_key : output_keys) {
      // check whether or not an input/output node
      if (::IsAllNum(output_key.c_str())) {
        fused_op_outputs[atoi(output_key.c_str())] = oedges.Get();
        outgoing_port = true;
      } else {
        const fusion_template::NodeDesc temp_output_node =
            temp_node_map[output_key];
        bool found = false;
        auto node_it = matched_node_map.find(temp_output_node.key);
        if (node_it != matched_node_map.end()) {
          // looping over all output edges bind to the same port
          found = oedges.CheckMatchedNode(temp_output_node, node_it->second,
                                          temp_node_map, matched_node_map,
                                          node_to_temp_key_, dy_mode);
        } else {
          // add new nodes
          found = oedges.AddNewNode(temp_output_node, output_key, temp_node_map,
                                    matched_node_map, node_to_temp_key_);
        }
        if (!found) {
          return false;
        }
      }
    }  // end for each consumer

    if (!outgoing_port && oedges.RemainEdge() > 0 &&
        node->type_string() != "Const") {
      // There's no cost to duplicate Const
      // has more consumers than the pattern
      return false;
    }
  }  // end for each output_port
  return true;
}

void FusionTemplate::AddInput(NodeDef& ndef, const Edge* iedge) {
  std::string input_name =
      absl::StrCat(iedge->src()->def().name(), ":", iedge->src_output());
  ndef.add_input(input_name);
}

void FusionTemplate::CopyAttr(NodeDef& dst, const NodeDef& src) {
  auto attr = src.attr();
  for (auto it = attr.begin(); it != attr.end(); ++it) {
    dst.mutable_attr()->insert({it->first, it->second});
  }
}

void FusionTemplate::AddInputEdge(Graph* g, Node* dst, const int dst_input,
                                  const Edge* ori_edge, const bool remove) {
  g->AddEdge(ori_edge->src(), ori_edge->src_output(), dst, dst_input);
  if (remove) {
    g->RemoveEdge(ori_edge);
  }
}

void FusionTemplate::AddOutputEdges(Graph* g, Node* src, const int src_output,
                                    std::vector<const Edge*>& ori_edges) {
  for (auto* ori_edge : ori_edges) {
    if (ori_edge != nullptr && ori_edge->dst() != nullptr) {
      g->AddEdge(src, src_output, ori_edge->dst(), ori_edge->dst_input());
      g->RemoveEdge(ori_edge);
    }
  }
}

void FusionTemplate::RemoveOutputEdges(Graph* g,
                                       std::vector<const Edge*>& ori_edges) {
  for (auto* ori_edge : ori_edges) {
    g->RemoveEdge(ori_edge);
  }
}

class TemplateBasedFusionImpl {
 public:
  explicit TemplateBasedFusionImpl(Graph* g, FusionTemplate* t);
  bool Fuse();

 private:
  bool VisitMatchedNodes();
  bool CheckOutputs(const Node* node,
                    const fusion_template::NodeDesc* temp_node);
  bool CheckInputs(const Node* node,
                   const fusion_template::NodeDesc* temp_node);
  bool CheckMatchedNodeInSameFrame();

 private:
  Graph* g_;
  FusionTemplate* t_;
  std::map<const std::string, fusion_template::NodeDesc> temp_node_map_;
  std::vector<const Edge*> fused_op_inputs_;
  std::vector<const Edge*> fused_op_deps_inputs_;
  std::vector<std::vector<const Edge*>> fused_op_outputs_;
  std::map<std::string, fusion_template::NodeMatching> matched_node_map_;
  int num_matched_;
  // for dynamic outputs of templates
  bool use_dynamic_output_keys_;
  bool use_dynamic_input_keys_;
  int dynamic_output_port_cur_;
  int dynamic_input_port_cur_;
  std::vector<std::vector<const Edge*>> fused_op_outputs_dynamic_;
  std::vector<const Edge*> fused_op_input_dynamic_;
  std::map<const Node*, std::string> node_frame_map_;
};

TemplateBasedFusionImpl::TemplateBasedFusionImpl(Graph* g, FusionTemplate* t)
    : g_(g), t_(t), num_matched_(0) {
  for (auto node : t_->temp_nodes_) {
    temp_node_map_.emplace(node.key, node);
  }
  fused_op_inputs_.resize(t_->num_inputs_);
  fused_op_outputs_.resize(t_->num_outputs_);
  use_dynamic_output_keys_ = false;
  use_dynamic_input_keys_ = false;

  std::unordered_set<Node*> enter_nodes;
  for (Node* node : g->nodes()) {
    node_frame_map_[node] = "";
    if (node->IsEnter()) {
      enter_nodes.insert(node);
    }
  }

  std::unordered_set<Node*> has_visited;
  for (Node* node : enter_nodes) {
    const std::string frame_name = node->def().attr().at("frame_name").s();
    std::queue<Node*> q;
    q.push(node);
    while (!q.empty()) {
      Node* n = q.front();
      q.pop();
      has_visited.insert(n);
      node_frame_map_[n] = frame_name;
      for (auto e : n->out_edges()) {
        Node* dst = e->dst();
        if (has_visited.find(dst) == has_visited.end() &&
            (!dst->IsExit() || !dst->IsNextIteration())) {
          q.push(dst);
        }
      }
    }
  }
}

bool TemplateBasedFusionImpl::VisitMatchedNodes() {
  bool all_visited = false;
  while (!all_visited) {
    all_visited = true;
    for (auto iter = matched_node_map_.begin(); iter != matched_node_map_.end();
         ++iter) {
      if (iter->second.visited) {
        continue;
      }
      all_visited = false;
      // check dynamic inputs
      auto search_itr =
          t_->nodes_dynamic_iedges_.find(temp_node_map_[iter->first].key);
      if (search_itr != t_->nodes_dynamic_iedges_.end()) {
        // inputs of this node is marked as dynamic
        if (!t_->CheckDynamicInputs(iter->second.node,
                                    &temp_node_map_[iter->first],
                                    search_itr->second, fused_op_inputs_,
                                    temp_node_map_, matched_node_map_)) {
          return false;
        }
      } else {
        // standard input checking
        if (!CheckInputs(iter->second.node, &temp_node_map_[iter->first])) {
          return false;
        }
      }
      // check dynamic outputs
      search_itr =
          t_->nodes_dynamic_oedges_.find(temp_node_map_[iter->first].key);
      if (search_itr != t_->nodes_dynamic_oedges_.end()) {
        // outputs of this node is marked as dynamic
        if (!t_->CheckDynamicOutputs(iter->second.node,
                                     &temp_node_map_[iter->first],
                                     search_itr->second, fused_op_outputs_,
                                     temp_node_map_, matched_node_map_)) {
          return false;
        }
      } else {
        // standard output checking
        if (!CheckOutputs(iter->second.node, &temp_node_map_[iter->first])) {
          return false;
        }
      }
      iter->second.visited = true;
    }
  }
  return true;
}

bool TemplateBasedFusionImpl::CheckOutputs(
    const Node* node, const fusion_template::NodeDesc* temp_node) {
  std::map<int, std::vector<const Edge*>> oedge_map;
  for (auto* oedge : node->out_edges()) {
    int output_port = oedge->src_output();
    oedge_map[output_port].push_back(oedge);
  }
  for (auto iter = oedge_map.begin(); iter != oedge_map.end(); ++iter) {
    int output_port =
        use_dynamic_output_keys_ ? dynamic_output_port_cur_ : iter->first;
    dynamic_output_port_cur_ = output_port;
    std::vector<const Edge*> oedges = iter->second;
    std::vector<std::string> output_keys;
    if (output_port == -1) {
      output_keys = temp_node->deps_outputs;
    } else {
      output_keys = temp_node->outputs[output_port];
    }
    bool outgoing_port = false;
    for (auto& output_key : output_keys) {
      if (::IsAllNum(output_key.c_str())) {
        fused_op_outputs_[atoi(output_key.c_str())] = iter->second;
        if (oedges.size() > 0) {
          oedges.erase(oedges.begin());
        }
        outgoing_port = true;
      } else if (output_key == "*") {
        // a case of dynamic outputs
        fused_op_outputs_dynamic_.push_back(iter->second);
        use_dynamic_output_keys_ = true;
        outgoing_port = true;
      } else {
        const fusion_template::NodeDesc temp_output_node =
            temp_node_map_[output_key];
        bool found = false;
        auto node_it = matched_node_map_.find(temp_output_node.key);
        if (node_it != matched_node_map_.end()) {
          for (auto oedge_it = oedges.begin(); oedge_it != oedges.end();
               ++oedge_it) {
            const Edge* oedge = *oedge_it;
            const Node* output_node = oedge->dst();
            if (output_node == node_it->second.node) {
              found = true;
              oedges.erase(oedge_it);
              break;
            }
          }
        } else {
          for (auto oedge_it = oedges.begin(); oedge_it != oedges.end();
               ++oedge_it) {
            const Edge* oedge = *oedge_it;
            const Node* output_node = oedge->dst();
            if (output_node->type_string() == temp_output_node.op) {
              fusion_template::NodeMatching matched_node(output_node);
              matched_node_map_.emplace(temp_output_node.key, matched_node);
              found = true;
              oedges.erase(oedge_it);
              break;
            }
          }
        }
        if (!found) {
          VLOG(2) << "Cant' find:" << temp_output_node.key
                  << ", op type:" << temp_output_node.op;
          return false;
        }
      }
    }  // end for each consumer
    if (!outgoing_port && oedges.size() > 0 &&
        node->type_string() != "Const") {  // There's no cost to duplicate Const
      // has more consumers than the pattern
      return false;
    }
  }  // end for each output_port
  use_dynamic_output_keys_ = false;
  return true;
}

bool TemplateBasedFusionImpl::CheckInputs(
    const Node* node, const fusion_template::NodeDesc* temp_node) {
  // require a sorting of in_edges by ascending order
  std::vector<std::tuple<int, const Edge*>> sorting_in_edges;
  ::sortInEdges(node->in_edges(), sorting_in_edges);
  std::set<std::string> visited_control_deps;
  std::vector<std::string> temp_deps_input_keys = temp_node->deps_inputs;
  auto deps_input_it = temp_deps_input_keys.begin();

  for (auto pair : sorting_in_edges) {
    auto* iedge = std::get<1>(pair);
    // added for dynamic input edges
    int input_port =
        use_dynamic_input_keys_ ? dynamic_input_port_cur_ : iedge->dst_input();
    dynamic_input_port_cur_ = input_port;
    if (input_port < 0) {
      if (node->type_string() == "Const" ||
          (node->type_string() == "Identity" && temp_deps_input_keys.empty())) {
        // TODO(minmin) not 100% sure about the safty of ignoring control
        // input of Const node. Best to avoid Const node in the Template
        VLOG(2) << "unexpected here:" << node->DebugString();
        continue;
      }
    }
    if (input_port >= (int)temp_node->inputs.size()) {
      LOG(FATAL) << "Please verify Template's node (" << node->type_string()
                 << ") definition"
                 << ", node inputs:" << node->in_edges().size()
                 << ", template node inputs:" << temp_node->inputs.size()
                 << " mismatch.";
    }
    const Node* input_node = iedge->src();
    // control dependency node
    if (input_port == -1) {
      bool found = false;
      if (temp_deps_input_keys.empty()) {
        VLOG(2) << "temp_deps_input_keys is empty"
                << ", and node type is:" << node->type_string();
        continue;
      }
      auto input_key = *deps_input_it;
      ++deps_input_it;
      if (::IsAllNum(input_key.c_str())) {
        fused_op_deps_inputs_.emplace_back(iedge);
        continue;
      } else {
        fusion_template::NodeDesc temp_input_node = temp_node_map_[input_key];
        if (input_node->type_string() == temp_input_node.op &&
            visited_control_deps.end() ==
                visited_control_deps.find(input_key)) {
          visited_control_deps.emplace(input_key);
          auto it = matched_node_map_.find(temp_input_node.key);
          if (it != matched_node_map_.end()) {
            if (input_node != it->second.node) {
              VLOG(2) << "port = -1 duplicate input:" << input_node->name()
                      << ", previous node:" << it->second.node->name();
              return false;
            }
          } else {
            fusion_template::NodeMatching matched_node(input_node);
            matched_node_map_.insert(
                std::make_pair(temp_input_node.key, matched_node));
          }
          found = true;
          break;
        }
        if (found) {
          continue;
        } else {
          VLOG(2) << "port = -1 not found input:" << input_node->name();
          return false;
        }
      }
    } else {
      std::string temp_input_key = temp_node->inputs[input_port];
      if (::IsAllNum(temp_input_key.c_str())) {
        fused_op_inputs_[atoi(temp_input_key.c_str())] = iedge;
        continue;
      }
      // added for dynamic input edges
      if (temp_input_key == "*") {
        // add to dynamic input edges
        fused_op_input_dynamic_.push_back(iedge);
        use_dynamic_input_keys_ = true;
        continue;
      } else {
        // turn off dynamic input when ever a non-dynamic edge appears
        // the dynamic template keys must be after the static template keys
        use_dynamic_input_keys_ = false;
      }

      const fusion_template::NodeDesc temp_input_node =
          temp_node_map_[temp_input_key];
      if (input_node->type_string() == temp_input_node.op) {
        auto it = matched_node_map_.find(temp_input_node.key);
        if (it != matched_node_map_.end()) {
          if (input_node != it->second.node) {
            VLOG(2) << "checkInput:" << temp_input_key
                    << ", input_node:" << input_node->name();
            return false;
          }
        } else {
          fusion_template::NodeMatching matched_node(input_node);
          matched_node_map_.insert(
              std::make_pair(temp_input_node.key, matched_node));
        }
        continue;
      }
      return false;
    }
  }
  use_dynamic_input_keys_ = false;
  return true;
}

bool TemplateBasedFusionImpl::CheckMatchedNodeInSameFrame() {
  // TODO: only op in default frame can be fused
  const Node* first_key_node = matched_node_map_[t_->first_key_].node;
  std::string frame_name = node_frame_map_[first_key_node];
  if (frame_name != "") return false;
  for (auto matched_node_it : matched_node_map_) {
    const Node* node = std::get<1>(matched_node_it).node;
    if (node_frame_map_[node] != frame_name) return false;
  }

  return true;
}

bool TemplateBasedFusionImpl::Fuse() {
  bool changed = false;
  // TODO(minmin) check Template consistency before really optimizing
  for (Node* node : g_->nodes()) {
    if (node->type_string() == temp_node_map_[t_->first_key_].op) {
      matched_node_map_.clear();
      t_->node_to_temp_key_.clear();
      fused_op_deps_inputs_.clear();
      fused_op_input_dynamic_.clear();
      fused_op_outputs_dynamic_.clear();
      fused_op_inputs_.resize(t_->num_inputs_);
      fused_op_outputs_.resize(t_->num_outputs_);
      for (int i = 0; i < fused_op_inputs_.size(); ++i) {
        fused_op_inputs_[i] = nullptr;
      }
      for (int i = 0; i < fused_op_outputs_.size(); ++i) {
        fused_op_outputs_[i].clear();
      }
      VLOG(3) << "try to match: " << t_->Name() << " " << node->name();
      VLOG(3) << "First Matched: " << node->name()
              << ", t->first_key:" << t_->first_key_
              << ", t->first_value:" << temp_node_map_[t_->first_key_].key;
      // check dynamic inputs
      auto search_itr =
          t_->nodes_dynamic_iedges_.find(temp_node_map_[t_->first_key_].key);
      if (search_itr != t_->nodes_dynamic_iedges_.end()) {
        if (!t_->CheckDynamicInputs(node, &temp_node_map_[t_->first_key_],
                                    search_itr->second, fused_op_inputs_,
                                    temp_node_map_, matched_node_map_)) {
          continue;
        }
      } else {
        if (!CheckInputs(node, &temp_node_map_[t_->first_key_])) {
          continue;
        }
      }
      // check dynamic outputs
      search_itr =
          t_->nodes_dynamic_oedges_.find(temp_node_map_[t_->first_key_].key);
      if (search_itr != t_->nodes_dynamic_oedges_.end()) {
        if (!t_->CheckDynamicOutputs(node, &temp_node_map_[t_->first_key_],
                                     search_itr->second, fused_op_outputs_,
                                     temp_node_map_, matched_node_map_)) {
          continue;
        }
      } else {
        if (!CheckOutputs(node, &temp_node_map_[t_->first_key_])) {
          continue;
        }
      }
      fusion_template::NodeMatching matched_node(node, true);
      matched_node_map_.insert(std::make_pair(t_->first_key_, matched_node));
      if (!VisitMatchedNodes()) {
        VLOG(2) << "VisitMatchedNodes failed";
        continue;
      }
      // double check the matched nodes
      if (matched_node_map_.size() != temp_node_map_.size()) {
        VLOG(2) << "Failed double check the matched nodes "
                << matched_node_map_.size() << " != " << temp_node_map_.size();
        continue;
      }

      // double check the matched nodes are in same frame
      if (!CheckMatchedNodeInSameFrame()) {
        VLOG(2) << "Failed double check the matched nodes, they are not in "
                   "same frame";
        continue;
      }
      // double check the matched inputs
      bool passed = true;
      for (int i = 0; i < t_->num_inputs_; ++i) {
        if (fused_op_inputs_[i] == nullptr) {
          passed = false;
          VLOG(2) << "failed check inputs";
          continue;
        }
      }
      if (!passed) {
        VLOG(2) << "Failed double check the matched inputs";
        continue;
      }

      if (fused_op_input_dynamic_.size() > 0) {
        // append dynamic in edges
        fused_op_inputs_.reserve(fused_op_inputs_.size() +
                                 fused_op_input_dynamic_.size());
        fused_op_inputs_.insert(fused_op_inputs_.end(),
                                fused_op_input_dynamic_.begin(),
                                fused_op_input_dynamic_.end());
      }

      // double check the matched outputs
      for (int i = 0; i < t_->num_outputs_; ++i) {
        if (fused_op_outputs_[i].empty()) {
          passed = false;
          continue;
        }
      }
      if (!passed) {
        VLOG(2) << "Failed double check the matched outputs";
        continue;
      }

      ++num_matched_;
      VLOG(3) << "Matched: " << num_matched_;
      for (auto iter = matched_node_map_.begin();
           iter != matched_node_map_.end(); ++iter) {
        VLOG(3) << "  " << iter->second.node->name();
      }

      std::string fused_op_name =
          absl::StrCat("__", num_matched_ - 1, "_fused");
      if (fused_op_outputs_dynamic_.size() > 0) {
        // append dynamic out edges
        fused_op_outputs_.reserve(fused_op_outputs_.size() +
                                  fused_op_outputs_dynamic_.size());
        fused_op_outputs_.insert(fused_op_outputs_.end(),
                                 fused_op_outputs_dynamic_.begin(),
                                 fused_op_outputs_dynamic_.end());
      }

      bool subgraph_replaced = false;
      if (t_->num_deps_inputs_ > 0) {
        subgraph_replaced = t_->AddSubgraph(
            matched_node_map_, fused_op_name, g_, fused_op_inputs_,
            fused_op_deps_inputs_, fused_op_outputs_);
      } else {
        subgraph_replaced =
            t_->AddSubgraph(matched_node_map_, fused_op_name, g_,
                            fused_op_inputs_, fused_op_outputs_);
      }

      VLOG(3) << "subgraph_replace:" << subgraph_replaced;

      if (!subgraph_replaced && t_->fused_op_ != "") {
        NodeDef* fused_def = new NodeDef();
        fused_def->set_op(t_->fused_op_);
        fused_def->set_name(fused_op_name);
        for (int i = 0; i < t_->num_inputs_; ++i) {
          const Edge* iedge = fused_op_inputs_[i];
          std::string input_name = absl::StrCat(iedge->src()->def().name(), ":",
                                                iedge->src_output());
          fused_def->add_input(input_name);
        }
        Status status;
        Node* fused_op = g_->AddNode(*fused_def, &status);
        if (status != Status::OK()) {
          VLOG(2) << status.error_message();
          continue;
        }
        for (int i = 0; i < t_->num_inputs_; ++i) {
          const Edge* iedge = fused_op_inputs_[i];
          g_->AddEdge(iedge->src(), iedge->src_output(), fused_op, i);
          g_->RemoveEdge(iedge);
        }
        for (int i = 0; i < t_->num_outputs_; ++i) {
          for (auto* oedge : fused_op_outputs_[i]) {
            g_->AddEdge(fused_op, i, oedge->dst(), oedge->dst_input());
            g_->RemoveEdge(oedge);
          }
        }
      }
      changed = true;
    }
  }
  if (num_matched_ > 0) {
    VLOG(1) << "Fused " << num_matched_ << " nodes as " << t_->Name()
            << " in graph " << static_cast<void*>(g_);
  }

  return changed;
}

bool Fuse(Graph* g, FusionTemplate& t) {
  TemplateBasedFusionImpl opt(g, &t);
  return opt.Fuse();
}
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

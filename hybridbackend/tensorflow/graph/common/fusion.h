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

#ifndef HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_FUSION_H_
#define HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_FUSION_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <absl/strings/str_cat.h>

#include <sys/types.h>
#include <algorithm>
#include <map>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

class Edge;
class Graph;
class Node;

namespace hybridbackend {

namespace fusion_template {

struct NodeDesc {
  std::string key;
  std::string op;
  std::vector<std::string> inputs;
  std::vector<std::vector<std::string>> outputs;
  std::vector<std::string> deps_inputs;
  std::vector<std::string> deps_outputs;
};

struct NodeMatching {
  explicit NodeMatching(const Node* n = nullptr, bool v = false) {
    node = n;
    visited = v;
    dy_offset = 1;
  }
  const Node* node;
  bool visited;
  // to specify the offset of dynamically replicated
  // graph nodes to the original one, start from 1 and has an
  // increment after each replication
  int dy_offset;
};
}  // namespace fusion_template

class FusionTemplate {
 public:
  FusionTemplate()
      : first_key_(""), num_inputs_(0), num_outputs_(0), fused_op_("") {}

  virtual bool AddSubgraph(
      std::map<std::string, fusion_template::NodeMatching>& nodes,
      std::string name_prefix, Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) = 0;

  virtual bool AddSubgraph(
      std::map<std::string, fusion_template::NodeMatching>& nodes,
      std::string name_prefix, Graph* g, std::vector<const Edge*>& inputs,
      std::vector<const Edge*>& deps_inputs,
      std::vector<std::vector<const Edge*>>& outputs) = 0;

  // check dynamic inputs
  // node: the target node in graph
  // temp_node: the target node in template
  // dy_mode: a flag to control dynamic node/edges processing
  // (1) dy_mode == 0, all dynamic input edges share the same endpoint
  // (2) dy_mode == 1, all dynamic input edges have their own distinct endpoint
  // (3) dy_mode == 2, no dynamic checking applied (override flag)
  virtual bool CheckDynamicInputs(
      const Node* node, const fusion_template::NodeDesc* temp_node,
      const int dy_mode, std::vector<const Edge*>& fused_op_inputs,
      std::map<const std::string, fusion_template::NodeDesc>& temp_node_map,
      std::map<std::string, fusion_template::NodeMatching>&
          matched_node_map) = 0;

  // check dynamic outputs
  // node: the target node in graph
  // temp_node: the target node in template
  // dy_mode: a flag to control dynamic node/edges processing
  // (1) dy_mode == 0, all dynamic output edges share the same endpoint
  // (2) dy_mode == 1, all dynamic output edges have their own distinct endpoint
  // (3) dy_mode == 2, no dynamic checking applied (override flag)
  virtual bool CheckDynamicOutputs(
      const Node* node, const fusion_template::NodeDesc* temp_node,
      const int dy_mode,
      std::vector<std::vector<const Edge*>>& fused_op_outputs,
      std::map<const std::string, fusion_template::NodeDesc>& temp_node_map,
      std::map<std::string, fusion_template::NodeMatching>&
          matched_node_map) = 0;

  virtual const string Name() = 0;

  std::vector<fusion_template::NodeDesc> temp_nodes_;
  std::string first_key_;
  int num_inputs_;
  int num_outputs_;
  int num_deps_inputs_ = 0;

  std::string fused_op_;
  // store nodes that has a dynamic number of in-edges from the same src node
  std::map<std::string, int> nodes_dynamic_iedges_;
  // store nodes that has a dynamic number of out-edges to the same dst node
  std::map<std::string, int> nodes_dynamic_oedges_;
  // store mapping from the name of an added node to its key in template
  std::map<std::string, std::string> node_to_temp_key_;
  std::vector<int> fused_op_input_idx_;
  std::vector<int> fused_op_output_idx_;

 protected:
  virtual bool CheckDynamicInputsImpl(
      const Node* node, const fusion_template::NodeDesc* temp_node,
      const int dy_mode, std::vector<const Edge*>& fused_op_inputs,
      std::map<const std::string, fusion_template::NodeDesc>& temp_node_map,
      std::map<std::string, fusion_template::NodeMatching>& matched_node_map,
      const int32 start_port, const int32 end_port);

  virtual bool CheckDynamicOutputsImpl(
      const Node* node, const fusion_template::NodeDesc* temp_node,
      const int dy_mode,
      std::vector<std::vector<const Edge*>>& fused_op_outputs,
      std::map<const std::string, fusion_template::NodeDesc>& temp_node_map,
      std::map<std::string, fusion_template::NodeMatching>& matched_node_map,
      const int32 start_port, const int32 end_port);

  // helper functions for constructing new subgraph
  void AddInput(NodeDef& ndef, const Edge* iedge);
  void CopyAttr(NodeDef& dst, const NodeDef& src);
  void AddInputEdge(Graph* g, Node* dst, const int dst_input,
                    const Edge* ori_edge, const bool remove = true);
  void AddOutputEdges(Graph* g, Node* src, const int src_output,
                      std::vector<const Edge*>& ori_edges);
  void RemoveOutputEdges(Graph* g, std::vector<const Edge*>& ori_edges);
};

// Returns true if and only if 'g' is mutated.
bool Fuse(Graph* g, FusionTemplate& t);

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW
#endif  // HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_FUSION_H_

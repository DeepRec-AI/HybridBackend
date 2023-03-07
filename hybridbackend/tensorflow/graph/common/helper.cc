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

#include "hybridbackend/tensorflow/graph/common/helper.h"
#include <absl/strings/str_join.h>

namespace tensorflow {
namespace hybridbackend {
namespace {
struct GraphNodeFormatter {
  void operator()(std::string* out, Node* n) const { out->append(n->name()); }
};
}  // namespace

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) < 1014L

template <typename T>
void DFSFromHelper(const Graph& g, gtl::ArraySlice<T> start,
                   const std::function<void(T)>& enter,
                   const std::function<void(T)>& leave,
                   const NodeComparator& stable_comparator,
                   const EdgeFilter& edge_filter) {
  // Stack of work to do.
  struct Work {
    T node;
    bool leave;  // Are we entering or leaving n?
  };
  std::vector<Work> stack(start.size());
  for (int i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }

  std::vector<bool> visited(g.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();

    T n = w.node;
    if (w.leave) {
      leave(n);
      continue;
    }

    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    if (enter) enter(n);

    // Arrange to call leave(n) when all done with descendants.
    if (leave) stack.push_back(Work{n, true});

    auto add_work = [&visited, &stack](Node* out) {
      if (!visited[out->id()]) {
        // Note; we must not mark as visited until we actually process it.
        stack.push_back(Work{out, false});
      }
    };

    if (stable_comparator) {
      std::vector<Node*> nodes_sorted;
      for (const Edge* out_edge : n->out_edges()) {
        if (!edge_filter || edge_filter(*out_edge)) {
          nodes_sorted.emplace_back(out_edge->dst());
        }
      }
      std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
      for (Node* out : nodes_sorted) {
        add_work(out);
      }
    } else {
      for (const Edge* out_edge : n->out_edges()) {
        if (!edge_filter || edge_filter(*out_edge)) {
          add_work(out_edge->dst());
        }
      }
    }
  }
}

void DFSFrom(const Graph& g, gtl::ArraySlice<Node*> start,
             const std::function<void(Node*)>& enter,
             const std::function<void(Node*)>& leave,
             const NodeComparator& stable_comparator,
             const EdgeFilter& edge_filter) {
  DFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}

#endif

string NodeJoin(const std::vector<Node*>& nodes, const string& delim) {
  return absl::StrJoin(nodes, delim, GraphNodeFormatter());
}

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

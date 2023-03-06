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

#ifndef HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_HELPER_H_
#define HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_HELPER_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <string>
#include <vector>

#include <tensorflow/core/graph/algorithm.h>
#include <tensorflow/core/graph/graph.h>
#include <tensorflow/core/public/version.h>

namespace tensorflow {
namespace hybridbackend {

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) < 1014L
template <typename T>
void DFSFromHelper(const Graph& g, gtl::ArraySlice<T> start,
                   const std::function<void(T)>& enter,
                   const std::function<void(T)>& leave,
                   const NodeComparator& stable_comparator,
                   const EdgeFilter& edge_filter);

void DFSFrom(const Graph& g, gtl::ArraySlice<Node*> start,
             const std::function<void(Node*)>& enter,
             const std::function<void(Node*)>& leave,
             const NodeComparator& stable_comparator = {},
             const EdgeFilter& edge_filter = {});
#endif

string NodeJoin(const std::vector<Node*>& nodes, const string& delim);

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW
#endif  // HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_HELPER_H_

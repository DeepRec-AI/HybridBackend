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

#ifndef HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_LINEARIZATION_H_
#define HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_LINEARIZATION_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <map>
#include <string>
#include <vector>

#include <tensorflow/core/graph/graph.h>

namespace tensorflow {
namespace hybridbackend {

class LinearizeOutputs {
 public:
  LinearizeOutputs(const string& op_type, const int32& op_output);
  Status In(Graph* graph);

 private:
  string op_type_;
  int32 op_output_;

  TF_DISALLOW_COPY_AND_ASSIGN(LinearizeOutputs);
};

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW
#endif  // HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_LINEARIZATION_H_

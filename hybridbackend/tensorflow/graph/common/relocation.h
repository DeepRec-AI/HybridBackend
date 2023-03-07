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

#ifndef HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_RELOCATION_H_
#define HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_RELOCATION_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <string>
#include <vector>

#include <tensorflow/core/graph/graph.h>

namespace tensorflow {
namespace hybridbackend {

class RelocateOutputs {
 public:
  RelocateOutputs(const string& op_type);
  RelocateOutputs& WithDevice(const string& device);
  RelocateOutputs& Force();
  Status In(Graph* graph);
  Status In(Graph* graph, int64* poccurrence_count);

 private:
  string op_type_;
  string device_;
  bool force_;

  TF_DISALLOW_COPY_AND_ASSIGN(RelocateOutputs);
};

class Relocate {
 public:
  Relocate(const string& op_type);
  Relocate& WithDevice(const string& device);
  Relocate& WithInput(const int32 input);
  Status In(Graph* graph);
  Status In(Graph* graph, int64* poccurrence_count);

 private:
  string op_type_;
  string device_;
  int32 input_;

  TF_DISALLOW_COPY_AND_ASSIGN(Relocate);
};

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW
#endif  // HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_RELOCATION_H_

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

#ifndef HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_REWRITING_H_
#define HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_REWRITING_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <string>
#include <vector>

#include <tensorflow/core/graph/graph.h>

namespace tensorflow {
namespace hybridbackend {

class Rewrite {
 public:
  Rewrite(const string& op_like_name, const string& op_name);
  Rewrite& WithDevice(const string& device);
  Rewrite& WithTypeAttr(const string& attr_name, const DataType& default_attr);
  Rewrite& WithShapeAttr(const string& attr_name,
                         const TensorShape& default_attr);
  Rewrite& WithIntAttr(const string& attr_name, const int32& default_attr);
  Rewrite& WithStrAttr(const string& attr_name, const string& default_attr);
  Rewrite& WithTypeListAttr(const string& attr_name);
  Status In(Graph* graph);
  Status In(Graph* graph, int64* poccurrence_count);

 private:
  string op_like_name_;
  string op_name_;
  string device_;
  int32 num_inputs_;
  std::map<string, DataType> type_attrs_;
  std::map<string, TensorShape> shape_attrs_;
  std::map<string, int32> int_attrs_;
  std::map<string, string> str_attrs_;
  std::vector<string> type_list_attrs_;

  TF_DISALLOW_COPY_AND_ASSIGN(Rewrite);
};

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW
#endif  // HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_REWRITING_H_

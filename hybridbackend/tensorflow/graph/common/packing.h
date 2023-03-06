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

#ifndef HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_PACKING_H_
#define HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_PACKING_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <map>
#include <string>
#include <vector>

#include <tensorflow/core/graph/graph.h>

namespace tensorflow {
namespace hybridbackend {

class Pack {
 public:
  Pack(const string& op_type, const string& optimized_op_type);
  Pack& WithDevice(const string& device);
  Pack& WithTypeAttr(const string& attr_name,
                     const std::vector<DataType>& constraints);
  Pack& WithShapeAttr(const string& attr_name);
  Pack& WithIntAttr(const string& attr_name);
  Pack& WithStrAttr(const string& attr_name);
  Pack& WithAggregatedShapeAttr(const string& attr_name);
  Pack& WithAggregatedIntAttr(const string& attr_name);
  Pack& WithAggregatedStrAttr(const string& attr_name);
  Pack& WithHandle(const int32 input);
  Pack& WithBuckets(const int32 num_buckets);
  Status In(Graph* graph);
  Status In(Graph* graph, int64* poccurrence_count);

 private:
  string op_type_;
  string optimized_op_type_;
  string device_;
  std::map<string, std::vector<DataType>> type_attrs_;
  std::vector<string> shape_attrs_;
  std::vector<string> int_attrs_;
  std::vector<string> str_attrs_;
  std::vector<string> aggregated_shape_attrs_;
  std::vector<string> aggregated_int_attrs_;
  std::vector<string> aggregated_str_attrs_;
  std::vector<int32> handles_;
  int num_buckets_;

  TF_DISALLOW_COPY_AND_ASSIGN(Pack);
};

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW
#endif  // HYBRIDBACKEND_TENSORFLOW_GRAPH_COMMON_PACKING_H_

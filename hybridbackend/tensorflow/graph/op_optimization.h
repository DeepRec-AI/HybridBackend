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

#ifndef HYBRIDBACKEND_TENSORFLOW_GRAPH_OP_OPTIMIZATION_H_
#define HYBRIDBACKEND_TENSORFLOW_GRAPH_OP_OPTIMIZATION_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <string>
#include <vector>

#include <tensorflow/core/common_runtime/optimization_registry.h>

namespace tensorflow {
namespace hybridbackend {

class OpOptimizationPass : public GraphOptimizationPass {
 public:
  virtual Status Run(const GraphOptimizationPassOptions& options);

 protected:
  virtual Status Optimize(Graph* graph, const SessionOptions* options,
                          const bool disabled) = 0;
};

#define REGISTER_REPLACING_OPTIMIZATION(PASS) \
  REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1, PASS)

#define REGISTER_REDUCTION_OPTIMIZATION(PASS)                                 \
  REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 100, \
                        PASS)

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW
#endif  // HYBRIDBACKEND_TENSORFLOW_GRAPH_OP_OPTIMIZATION_H_

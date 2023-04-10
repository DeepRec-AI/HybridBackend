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

#include <vector>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/graph/common/linearization.h"
#include "hybridbackend/tensorflow/graph/common/packing.h"
#include "hybridbackend/tensorflow/graph/common/rewriting.h"
#include "hybridbackend/tensorflow/graph/op_optimization.h"

namespace tensorflow {
namespace hybridbackend {

class OptimizeLookupReplacingPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    TF_RETURN_IF_ERROR(Rewrite("Lookup", "HbLookup").In(graph));

    return Status::OK();
  }
};

REGISTER_REPLACING_OPTIMIZATION(OptimizeLookupReplacingPass);

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

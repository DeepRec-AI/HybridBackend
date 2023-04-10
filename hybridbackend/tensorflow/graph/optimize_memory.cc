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

#include <map>
#include <set>
#include <vector>

#include <stdlib.h>

#include <absl/strings/str_cat.h>
#include <tensorflow/core/graph/node_builder.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/graph/common/packing.h"
#include "hybridbackend/tensorflow/graph/common/rewriting.h"
#include "hybridbackend/tensorflow/graph/op_optimization.h"

namespace tensorflow {
namespace hybridbackend {

class OptimizeMemoryReplacingPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    if (TF_PREDICT_FALSE(disabled)) {
      return Status::OK();
    }

    ::hybridbackend::EnvVarSetIfNotExists("HB_MEMORY_DECAY_MILLIS", 60000);
    const int kMemoryDecayMillis =
        ::hybridbackend::EnvVarGetInt("HB_MEMORY_DECAY_MILLIS", 0);
    ::hybridbackend::EnvVarSetIfNotExists(
        "MALLOC_CONF", "background_thread:true,metadata_thp:auto");
    VLOG(1) << "Memory decay set to " << kMemoryDecayMillis << "ms";

    return Status::OK();
  }
};

REGISTER_REPLACING_OPTIMIZATION(OptimizeMemoryReplacingPass);

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

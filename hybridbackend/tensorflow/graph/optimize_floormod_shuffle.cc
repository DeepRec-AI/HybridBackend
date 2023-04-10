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
#include "hybridbackend/tensorflow/graph/common/packing.h"
#include "hybridbackend/tensorflow/graph/common/relocation.h"
#include "hybridbackend/tensorflow/graph/common/replacing.h"
#include "hybridbackend/tensorflow/graph/common/rewriting.h"
#include "hybridbackend/tensorflow/graph/op_optimization.h"

namespace tensorflow {
namespace hybridbackend {

namespace {
inline bool FloormodShuffleOptimizationDisabled() {
  static const bool kFloormodShuffleOptimizationDisabled =
      ::hybridbackend::EnvVarGetBool(
          "HB_OP_FLOORMOD_SHUFFLE_OPTIMIZATION_DISABLED", false);
  return kFloormodShuffleOptimizationDisabled;
}

inline bool FloormodShufflePackingDisabled() {
  static const bool kFloormodShufflePackingDisabled =
      ::hybridbackend::EnvVarGetBool("HB_OP_FLOORMOD_SHUFFLE_PACKING_DISABLED",
                                     false);
  return kFloormodShufflePackingDisabled;
}

}  // namespace

class OptimizeFloormodShuffleReplacingPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    if (TF_PREDICT_FALSE(disabled || FloormodShuffleOptimizationDisabled())) {
      return Status::OK();
    }

    TF_RETURN_IF_ERROR(Rewrite("FloormodShuffle", "HbFloormodShuffle")
                           .WithIntAttr("num_partitions", 1)
                           .In(graph));

    return Status::OK();
  }
};

REGISTER_REPLACING_OPTIMIZATION(OptimizeFloormodShuffleReplacingPass);

class OptimizeFloormodShuffleReductionPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    if (TF_PREDICT_FALSE(disabled || FloormodShuffleOptimizationDisabled())) {
      return Status::OK();
    }

    if (TF_PREDICT_TRUE(!FloormodShufflePackingDisabled())) {
      TF_RETURN_IF_ERROR(
          Pack("HbFloormodShuffle", "HbFloormodShuffleN")
              .WithTypeAttr("T", {DT_INT32, DT_INT64, DT_UINT32, DT_UINT64})
              .WithIntAttr("num_partitions")
              .In(graph));

      return Status::OK();
    }

    return Status::OK();
  }
};

REGISTER_REDUCTION_OPTIMIZATION(OptimizeFloormodShuffleReductionPass);
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

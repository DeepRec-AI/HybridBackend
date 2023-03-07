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
inline bool PartitionByModuloOptimizationDisabled() {
  static const bool kPartitionByModuloOptimizationDisabled =
      ::hybridbackend::EnvVarGetBool(
          "HB_OP_PARTITION_BY_MODULO_OPTIMIZATION_DISABLED", false);
  return kPartitionByModuloOptimizationDisabled;
}

inline bool PartitionByModuloPackingDisabled() {
  static const bool kPartitionByModuloPackingDisabled =
      ::hybridbackend::EnvVarGetBool(
          "HB_OP_PARTITION_BY_MODULO_PACKING_DISABLED", false);
  return kPartitionByModuloPackingDisabled;
}

}  // namespace

class OptimizePartitionByModuloReplacingPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    if (TF_PREDICT_FALSE(disabled || PartitionByModuloOptimizationDisabled())) {
      return Status::OK();
    }

    TF_RETURN_IF_ERROR(Rewrite("PartitionByModulo", "HbPartitionByModulo")
                           .WithIntAttr("num_partitions", 1)
                           .In(graph));

    return Status::OK();
  }
};

REGISTER_REPLACING_OPTIMIZATION(OptimizePartitionByModuloReplacingPass);

class OptimizePartitionByModuloReductionPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    if (TF_PREDICT_FALSE(disabled || PartitionByModuloOptimizationDisabled())) {
      return Status::OK();
    }

    if (TF_PREDICT_TRUE(!PartitionByModuloPackingDisabled())) {
      TF_RETURN_IF_ERROR(
          Pack("HbPartitionByModulo", "HbPartitionByModuloN")
              .WithTypeAttr("T", {DT_INT32, DT_INT64, DT_UINT32, DT_UINT64})
              .WithIntAttr("num_partitions")
              .In(graph));

      return Status::OK();
    }

    return Status::OK();
  }
};

REGISTER_REDUCTION_OPTIMIZATION(OptimizePartitionByModuloReductionPass);
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

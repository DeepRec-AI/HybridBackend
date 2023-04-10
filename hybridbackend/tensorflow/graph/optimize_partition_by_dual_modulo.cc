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
inline bool PartitionByDualModuloOptimizationDisabled() {
  static const bool kPartitionByDualModuloOptimizationDisabled =
      ::hybridbackend::EnvVarGetBool(
          "HB_OP_PARTITION_BY_DUAL_MODULO_OPTIMIZATION_DISABLED", false);
  return kPartitionByDualModuloOptimizationDisabled;
}

inline bool PartitionByDualModuloPackingDisabled() {
  static const bool kPartitionByDualModuloPackingDisabled =
      ::hybridbackend::EnvVarGetBool(
          "HB_OP_PARTITION_BY_DUAL_MODULO_PACKING_DISABLED", false);
  return kPartitionByDualModuloPackingDisabled;
}
}  // namespace

class OptimizePartitionByDualModuloReplacingPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    if (TF_PREDICT_FALSE(disabled ||
                         PartitionByDualModuloOptimizationDisabled())) {
      return Status::OK();
    }

    TF_RETURN_IF_ERROR(Rewrite("PartitionByDualModuloStageOne",
                               "HbPartitionByDualModuloStageOne")
                           .WithIntAttr("num_partitions", 1)
                           .WithIntAttr("modulus", 1)
                           .In(graph));

    TF_RETURN_IF_ERROR(Rewrite("PartitionByDualModuloStageTwo",
                               "HbPartitionByDualModuloStageTwo")
                           .WithIntAttr("num_partitions", 1)
                           .WithIntAttr("modulus", 1)
                           .In(graph));

    return Status::OK();
  }
};

REGISTER_REPLACING_OPTIMIZATION(OptimizePartitionByDualModuloReplacingPass);

class OptimizePartitionByDualModuloReductionPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    if (TF_PREDICT_FALSE(disabled ||
                         PartitionByDualModuloOptimizationDisabled())) {
      return Status::OK();
    }

    if (TF_PREDICT_TRUE(!PartitionByDualModuloPackingDisabled())) {
      TF_RETURN_IF_ERROR(
          Pack("HbPartitionByDualModuloStageOne",
               "HbPartitionByDualModuloStageOneN")
              .WithTypeAttr("T", {DT_INT32, DT_INT64, DT_UINT32, DT_UINT64})
              .WithIntAttr("num_partitions")
              .WithIntAttr("modulus")
              .In(graph));
      TF_RETURN_IF_ERROR(
          Pack("HbPartitionByDualModuloStageTwo",
               "HbPartitionByDualModuloStageTwoN")
              .WithTypeAttr("T", {DT_INT32, DT_INT64, DT_UINT32, DT_UINT64})
              .WithIntAttr("num_partitions")
              .WithIntAttr("modulus")
              .In(graph));
      return Status::OK();
    }

    return Status::OK();
  }
};

REGISTER_REDUCTION_OPTIMIZATION(OptimizePartitionByDualModuloReductionPass);
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

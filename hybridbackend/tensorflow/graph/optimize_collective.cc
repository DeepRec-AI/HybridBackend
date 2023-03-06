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

namespace {
inline bool CollectivePackingDisabled() {
  static const bool kCollectivePackingDisabled =
      ::hybridbackend::EnvVarGetBool("HB_COLLECTIVE_PACKING_DISABLED", false);
  return kCollectivePackingDisabled;
}
inline string CollectivePackingAlgorithm() {
  // MERGE or GROUP allowed
  static const string kCollectivePackingAlgorithm =
      ::hybridbackend::EnvVarGet("HB_COLLECTIVE_PACKING_ALGORITHM", "GROUP");
  return kCollectivePackingAlgorithm;
}
}  // namespace

class OptimizeCollectiveReplacingPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    TF_RETURN_IF_ERROR(Rewrite("GetCollectiveId", "HbGetNcclId").In(graph));
    TF_RETURN_IF_ERROR(Rewrite("CollectiveHandleOp", "HbNcclCollectiveHandleOp")
                           .WithStrAttr("container", "")
                           .WithStrAttr("shared_name", "")
                           .In(graph));
    TF_RETURN_IF_ERROR(Rewrite("CreateCollective", "HbCreateNcclCollective")
                           .WithStrAttr("shared_name", "")
                           .WithIntAttr("world_size", 1)
                           .WithIntAttr("local_size", 1)
                           .WithIntAttr("rank", 0)
                           .In(graph));
    TF_RETURN_IF_ERROR(
        Rewrite("IsCollectiveInitialized", "HbIsNcclCollectiveInitialized")
            .In(graph));

    // Attribute T should be infered not specified.
    TF_RETURN_IF_ERROR(Rewrite("CollectiveAllreduce", "HbNcclAllreduce")
                           .WithIntAttr("reduce_op", 0)
                           .In(graph));
    // Attribute T should be infered not specified.
    TF_RETURN_IF_ERROR(Rewrite("CollectiveAlltoall", "HbNcclAlltoall")
                           .WithIntAttr("topology", 0)
                           .WithTypeAttr("wire_dtype", DT_FLOAT)
                           .In(graph));
    // Attribute T should be infered not specified.
    TF_RETURN_IF_ERROR(Rewrite("CollectiveAlltoallv", "HbNcclAlltoallv")
                           .WithIntAttr("topology", 0)
                           .WithTypeAttr("wire_dtype", DT_FLOAT)
                           .WithShapeAttr("common_shape", TensorShape({}))
                           .In(graph));
    // Attribute T should be infered not specified.
    TF_RETURN_IF_ERROR(Rewrite("CollectiveBroadcast", "HbNcclBroadcast")
                           .WithIntAttr("root_rank", 0)
                           .In(graph));
    // Attribute T should be infered not specified.
    TF_RETURN_IF_ERROR(
        Rewrite("CollectiveAllgather", "HbNcclAllgather").In(graph));
    // Attribute T should be infered not specified.
    TF_RETURN_IF_ERROR(
        Rewrite("CollectiveAllgatherv", "HbNcclAllgatherv").In(graph));

    return Status::OK();
  }
};

REGISTER_REPLACING_OPTIMIZATION(OptimizeCollectiveReplacingPass);

class OptimizeCollectiveReductionPass : public OpOptimizationPass {
 public:
  Status Optimize(Graph* graph, const SessionOptions* options,
                  const bool disabled) override {
    if (TF_PREDICT_TRUE(!disabled && !CollectivePackingDisabled())) {
      const std::vector<DataType> nccl_dtypes{DT_FLOAT, DT_DOUBLE, DT_HALF,
                                              DT_INT8,  DT_INT32,  DT_INT64,
                                              DT_UINT8, DT_UINT32, DT_INT64};
      const std::vector<DataType> nccl_wire_dtypes{DT_FLOAT, DT_HALF};
      if (TF_PREDICT_FALSE(CollectivePackingAlgorithm() == "GROUP")) {
        TF_RETURN_IF_ERROR(Pack("HbNcclAllreduce", "HbNcclAllreduceN")
                               .WithTypeAttr("dtype", nccl_dtypes)
                               .WithIntAttr("reduce_op")
                               .WithHandle(0)
                               .In(graph));
        TF_RETURN_IF_ERROR(Pack("HbNcclAlltoall", "HbNcclAlltoallN")
                               .WithTypeAttr("dtype", nccl_dtypes)
                               .WithTypeAttr("wire_dtype", nccl_wire_dtypes)
                               .WithHandle(0)
                               .In(graph));
        TF_RETURN_IF_ERROR(Pack("HbNcclAlltoallv", "HbNcclAlltoallvN")
                               .WithTypeAttr("dtype", nccl_dtypes)
                               .WithTypeAttr("wire_dtype", nccl_wire_dtypes)
                               .WithAggregatedShapeAttr("common_shape")
                               .WithHandle(0)
                               .In(graph));
      } else if (TF_PREDICT_FALSE(CollectivePackingAlgorithm() == "MERGE")) {
        TF_RETURN_IF_ERROR(Pack("HbNcclAllreduce", "HbNcclAllreduceMergedN")
                               .WithTypeAttr("dtype", nccl_dtypes)
                               .WithIntAttr("reduce_op")
                               .WithHandle(0)
                               .In(graph));
        TF_RETURN_IF_ERROR(Pack("HbNcclAlltoall", "HbNcclAlltoallMergedN")
                               .WithTypeAttr("dtype", nccl_dtypes)
                               .WithTypeAttr("wire_dtype", nccl_wire_dtypes)
                               .WithHandle(0)
                               .In(graph));
        TF_RETURN_IF_ERROR(Pack("HbNcclAlltoallv", "HbNcclAlltoallvMergedN")
                               .WithTypeAttr("dtype", nccl_dtypes)
                               .WithTypeAttr("wire_dtype", nccl_wire_dtypes)
                               .WithShapeAttr("common_shape")
                               .WithHandle(0)
                               .In(graph));
      } else {
        TF_RETURN_IF_ERROR(Pack("HbNcclAllreduce", "HbNcclAllreduceMergedN")
                               .WithTypeAttr("dtype", nccl_dtypes)
                               .WithIntAttr("reduce_op")
                               .WithHandle(0)
                               .In(graph));
        TF_RETURN_IF_ERROR(Pack("HbNcclAlltoall", "HbNcclAlltoallMergedN")
                               .WithTypeAttr("dtype", nccl_dtypes)
                               .WithTypeAttr("wire_dtype", nccl_wire_dtypes)
                               .WithHandle(0)
                               .In(graph));
        TF_RETURN_IF_ERROR(Pack("HbNcclAlltoallv", "HbNcclAlltoallvN")
                               .WithTypeAttr("dtype", nccl_dtypes)
                               .WithTypeAttr("wire_dtype", nccl_wire_dtypes)
                               .WithAggregatedShapeAttr("common_shape")
                               .WithHandle(0)
                               .In(graph));
      }
    }

    TF_RETURN_IF_ERROR(
        LinearizeOutputs("HbNcclCollectiveHandleOp", 0).In(graph));
    return Status::OK();
  }
};

REGISTER_REDUCTION_OPTIMIZATION(OptimizeCollectiveReductionPass);

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

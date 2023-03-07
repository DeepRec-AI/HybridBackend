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

#include <unistd.h>
#include <fstream>
#include <vector>

#include <absl/strings/str_cat.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/graph/common/relocation.h"
#include "hybridbackend/tensorflow/graph/op_optimization.h"

namespace tensorflow {
namespace hybridbackend {

namespace {
inline bool OpOptimizationDisabled() {
  static const bool kOptimizationDisabled =
      ::hybridbackend::EnvVarGetBool("HB_OP_OPTIMIZATION_DISABLED", false);
  return kOptimizationDisabled;
}

inline bool OpRelocationEnabled() {
  static const bool kRelocationEnabled =
      ::hybridbackend::EnvVarGetBool("HB_OP_RELOCATION_ENABLED", false);
  return kRelocationEnabled;
}
}  // namespace

Status OpOptimizationPass::Run(const GraphOptimizationPassOptions& options) {
  if (options.graph == nullptr) {
    return Status::OK();
  }

  Graph* graph = options.graph->get();
  if (graph == nullptr) {
    return errors::Internal(
        "Op optimization should happen before partitioning "
        "and a graph should be available.");
  }

  bool opt_disabled = false;

#if HYBRIDBACKEND_CHECK_INSTANCE
  static long kTwoSeconds = 2000L;
  static bool kInstanceChecked = ::hybridbackend::EnvCheckInstance(kTwoSeconds);
  if (TF_PREDICT_FALSE(!kInstanceChecked)) {
    opt_disabled = true;
  }
#endif

  opt_disabled |= OpOptimizationDisabled();

  TF_RETURN_IF_ERROR(Optimize(graph, options.session_options, opt_disabled));
  return Status::OK();
}

class OpOptimizationBeginPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.graph == nullptr) {
      return Status::OK();
    }

    Graph* graph = options.graph->get();
    if (graph == nullptr) {
      return errors::Internal(
          "Op optimization should happen before partitioning "
          "and a graph should be available.");
    }

    static const string kUnoptimizedGraphPath =
        ::hybridbackend::EnvVarGet("HB_UNOPTIMIZED_GRAPH_PATH", "");

    if (!kUnoptimizedGraphPath.empty()) {
      std::ofstream ofs(absl::StrCat(kUnoptimizedGraphPath, ".", getpid(),
                                     "_0x", absl::Hex(graph), ".pbtxt"));
      ofs << graph->ToGraphDefDebug().DebugString();
    }

    if (TF_PREDICT_FALSE(OpRelocationEnabled())) {
      RelocateOutputs("Unique").Force().In(graph);
    }

    return Status::OK();
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0,
                      OpOptimizationBeginPass);

class OpOptimizationEndPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.graph == nullptr) {
      return Status::OK();
    }

    Graph* graph = options.graph->get();
    if (graph == nullptr) {
      return errors::Internal(
          "Op optimization should happen before partitioning "
          "and a graph should be available.");
    }

    static const string kOptimizedGraphPath =
        ::hybridbackend::EnvVarGet("HB_OPTIMIZED_GRAPH_PATH", "");

    if (!kOptimizedGraphPath.empty()) {
      std::ofstream ofs(absl::StrCat(kOptimizedGraphPath, ".", getpid(), "_0x",
                                     absl::Hex(graph), ".pbtxt"));
      ofs << graph->ToGraphDefDebug().DebugString();
    }

    return Status::OK();
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 1000,
                      OpOptimizationEndPass);

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

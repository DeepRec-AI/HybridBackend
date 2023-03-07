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

#include <string>
#include <vector>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/util/device_name_utils.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/graph/common/helper.h"
#include "hybridbackend/tensorflow/graph/common/relocation.h"

namespace tensorflow {
namespace hybridbackend {

RelocateOutputs::RelocateOutputs(const string& op_type_)
    : op_type_(op_type_), device_(DEVICE_GPU) {}

RelocateOutputs& RelocateOutputs::WithDevice(const string& device) {
  device_ = device;
  return *this;
}

RelocateOutputs& RelocateOutputs::Force() {
  force_ = true;
  return *this;
}

Status RelocateOutputs::In(Graph* graph) { return In(graph, nullptr); }

Status RelocateOutputs::In(Graph* graph, int64* poccurrence_count) {
  int64 occurrence_count = 0;
  for (Node* node : graph->op_nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    if (node->type_string() != op_type_) {
      continue;
    }

    DeviceNameUtils::ParsedName root_device;
    if (!DeviceNameUtils::ParseFullName(node->requested_device(),
                                        &root_device)) {
      continue;
    }

    DFSFrom(
        *graph, {node},
        [&](Node* n) {
          if (!n->IsOp()) {
            return;
          }

          string original_device = n->requested_device();
          DeviceNameUtils::ParsedName requested_device;
          if (!DeviceNameUtils::ParseFullName(original_device,
                                              &requested_device)) {
            return;
          }

          if (requested_device.job != root_device.job ||
              requested_device.replica != root_device.replica ||
              requested_device.task != root_device.task) {
            VLOG(1) << "Skipped relocation of " << n->name()
                    << " since task differs";
            return;
          }

          if (requested_device.has_type) {
            if (requested_device.type == device_) {
              return;
            }
            if (!force_) {
              VLOG(1) << "Skipped relocation of " << n->name()
                      << " since device already set: " << requested_device.type;
              return;
            }
          }

          static constexpr char kResourceAttrName[] = "container";

          if (HasNodeAttr(n->def(), kResourceAttrName)) {
            VLOG(1) << "Skipped relocation of " << n->name()
                    << " since resource in graph " << static_cast<void*>(graph);
            return;
          }

          static constexpr char kColocationAttrName[] = "_class";
          if (HasNodeAttr(n->def(), kColocationAttrName)) {
            VLOG(1) << "Skipped relocation of " << n->name()
                    << " since colocation in graph "
                    << static_cast<void*>(graph);
            return;
          }

          requested_device.type = device_;
          requested_device.has_type = true;

          auto relocated_device =
              DeviceNameUtils::ParsedNameToString(requested_device);
          n->set_requested_device(relocated_device);
          VLOG(1) << "Relocated " << n->name() << " from \"" << original_device
                  << "\" to \"" << relocated_device << "\" (" << device_
                  << ") in graph " << static_cast<void*>(graph);
          occurrence_count++;
        },
        nullptr);
  }

  if (poccurrence_count != nullptr) {
    *poccurrence_count = occurrence_count;
  }
  return Status::OK();
}

Relocate::Relocate(const string& op_type_)
    : op_type_(op_type_), device_(DEVICE_GPU), input_(0) {}

Relocate& Relocate::WithDevice(const string& device) {
  device_ = device;
  return *this;
}

Relocate& Relocate::WithInput(const int32 input) {
  input_ = input;
  return *this;
}

Status Relocate::In(Graph* graph) { return In(graph, nullptr); }

Status Relocate::In(Graph* graph, int64* poccurrence_count) {
  int64 occurrence_count = 0;
  for (Node* node : graph->op_nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    if (node->type_string() != op_type_) {
      continue;
    }

    const Edge* edge;
    auto status = node->input_edge(input_, &edge);
    if (!status.ok()) {
      continue;
    }

    DeviceNameUtils::ParsedName requested_device;
    if (!DeviceNameUtils::ParseFullName(node->requested_device(),
                                        &requested_device)) {
      continue;
    }

    if (requested_device.type == device_) {
      continue;
    }

    static constexpr char kColocationAttrName[] = "_class";

    if (node->requested_device() != edge->src()->requested_device()) {
      if (HasNodeAttr(node->def(), kColocationAttrName)) {
        node->ClearAttr(kColocationAttrName);
      }
      node->set_requested_device(edge->src()->requested_device());
      VLOG(1) << "Relocated " << node->name() << " with input "
              << edge->src()->name() << " in graph "
              << static_cast<void*>(graph);
      occurrence_count++;
    }
  }

  if (poccurrence_count != nullptr) {
    *poccurrence_count = occurrence_count;
  }
  return Status::OK();
}

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

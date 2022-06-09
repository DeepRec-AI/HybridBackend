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

#include "hybridbackend/common/profiler.h"

namespace hybridbackend {

ProfilerRange* ProfilerRange::forSynch(const std::string& message) {
#if HYBRIDBACKEND_NVTX
  return new ProfilerRange("Synch Ops", message.c_str());
#else
  return nullptr;
#endif
}

ProfilerRange* ProfilerRange::forLookup(const std::string& message) {
#if HYBRIDBACKEND_NVTX
  return new ProfilerRange("Lookup Ops", message.c_str());
#else
  return nullptr;
#endif
}

ProfilerRange::ProfilerRange(const std::string& domain,
                             const std::string& message) {
#if HYBRIDBACKEND_NVTX
  domain_ = nvtxDomainCreateA(domain.c_str());
  nvtxEventAttributes_t nvtx_attr = {0};
  nvtx_attr.version = NVTX_VERSION;
  nvtx_attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  nvtx_attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_attr.message.ascii = message.c_str();
  range_ = nvtxDomainRangeStartEx(domain_, &nvtx_attr);
#endif
}

ProfilerRange::~ProfilerRange() {
#if HYBRIDBACKEND_NVTX
  nvtxDomainRangeEnd(domain_, range_);
#endif
}

}  // namespace hybridbackend

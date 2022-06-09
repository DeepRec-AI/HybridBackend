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

#ifndef HYBRIDBACKEND_COMMON_PROFILER_H_
#define HYBRIDBACKEND_COMMON_PROFILER_H_

#include <string>

#if HYBRIDBACKEND_NVTX
#include <nvToolsExt.h>
#endif

namespace hybridbackend {

class ProfilerRange {
 public:
  static ProfilerRange* forSynch(const std::string& message);
  static ProfilerRange* forLookup(const std::string& message);

  ProfilerRange(const std::string& domain, const std::string& message);
  ~ProfilerRange();

 private:
#if HYBRIDBACKEND_NVTX
  nvtxDomainHandle_t domain_;
  nvtxRangeId_t range_;
#endif
};

}  // namespace hybridbackend

#endif  // HYBRIDBACKEND_COMMON_PROFILER_H_

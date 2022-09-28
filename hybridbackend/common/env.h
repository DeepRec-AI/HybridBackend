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

#ifndef HYBRIDBACKEND_COMMON_ENV_H_
#define HYBRIDBACKEND_COMMON_ENV_H_

#include <string>

namespace hybridbackend {

std::string EnvVarGet(const std::string& env_var,
                      const std::string& default_val);

int EnvVarGetInt(const std::string& env_var, const int default_val);

bool EnvVarGetBool(const std::string& env_var, const bool default_val);

std::string EnvHttpGet(const std::string& url, const std::string& default_val,
                       const long timeout);

int EnvHttpGetInt(const std::string& url, const int default_val,
                  const long timeout);

bool EnvCheckInstance(const long timeout);

int EnvGetGpuCount();

}  // namespace hybridbackend

#endif  // HYBRIDBACKEND_COMMON_ENV_H_

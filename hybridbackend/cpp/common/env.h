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

#ifndef HYBRIDBACKEND_CPP_COMMON_ENV_H_
#define HYBRIDBACKEND_CPP_COMMON_ENV_H_

#include <string>

#ifdef __has_builtin
#define HYBRIDBACKEND_HAS_BUILTIN(x) __has_builtin(x)
#else
#define HYBRIDBACKEND_HAS_BUILTIN(x) 0
#endif

#if (!defined(__NVCC__)) && (HYBRIDBACKEND_HAS_BUILTIN(__builtin_expect) || \
                             (defined(__GNUC__) && __GNUC__ >= 3))
#define HYBRIDBACKEND_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define HYBRIDBACKEND_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define HYBRIDBACKEND_PREDICT_FALSE(x) (x)
#define HYBRIDBACKEND_PREDICT_TRUE(x) (x)
#endif

namespace hybridbackend {

int EnvGetInt(const std::string& env_var, int default_val);

std::string EnvGet(const std::string& env_var, const std::string& default_val);

}  // namespace hybridbackend

#endif  // HYBRIDBACKEND_CPP_COMMON_ENV_H_

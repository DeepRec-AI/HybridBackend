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

#ifndef HYBRIDBACKEND_COMMON_MACROS_H_
#define HYBRIDBACKEND_COMMON_MACROS_H_

#ifdef __has_builtin
#define HB_HAS_BUILTIN(x) __has_builtin(x)
#else
#define HB_HAS_BUILTIN(x) 0
#endif

#if (!defined(__NVCC__)) && \
    (HB_HAS_BUILTIN(__builtin_expect) || (defined(__GNUC__) && __GNUC__ >= 3))
#define HB_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define HB_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define HB_PREDICT_FALSE(x) (x)
#define HB_PREDICT_TRUE(x) (x)
#endif

#define HB_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete

#endif  // HYBRIDBACKEND_COMMON_MACROS_H_

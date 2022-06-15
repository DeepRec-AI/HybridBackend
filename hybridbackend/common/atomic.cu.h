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

#ifndef HYBRIDBACKEND_COMMON_ATOMIC_CU_H_
#define HYBRIDBACKEND_COMMON_ATOMIC_CU_H_

__forceinline__ __device__ long atomicAdd(long* address, long val) {
  return (long)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__forceinline__ __device__ long long atomicAdd(long long* address,
                                               long long val) {
  return (long long)atomicAdd((unsigned long long*)address,
                              (unsigned long long)val);
}

__forceinline__ __device__ unsigned long atomicAdd(unsigned long* address,
                                                   unsigned long val) {
  return (unsigned long)atomicAdd((unsigned long long*)address,
                                  (unsigned long long)val);
}

__forceinline__ __device__ long atomicCAS(long* address, long compare,
                                          long val) {
  return (long)atomicCAS((unsigned long long*)address,
                         (unsigned long long)compare, (unsigned long long)val);
}

__forceinline__ __device__ long long atomicCAS(long long* address,
                                               long long compare,
                                               long long val) {
  return (long long)atomicCAS((unsigned long long*)address,
                              (unsigned long long)compare,
                              (unsigned long long)val);
}

__forceinline__ __device__ unsigned long atomicCAS(unsigned long* address,
                                                   unsigned long compare,
                                                   unsigned long val) {
  return (unsigned long)atomicCAS((unsigned long long*)address,
                                  (unsigned long long)compare,
                                  (unsigned long long)val);
}

#endif  // HYBRIDBACKEND_COMMON_ATOMIC_CU_H_

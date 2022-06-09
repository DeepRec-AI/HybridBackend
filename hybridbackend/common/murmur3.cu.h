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

MurmurHash3 was written by Austin Appleby, and is placed in the public
domain. The author hereby disclaims copyright to this source code.
Note - The x86 and x64 versions do _not_ produce the same results, as the
algorithms are optimized for their respective platforms. You can still
compile and run any of them on any platform, but your performance with the
non-native version will be less than optimal.

See https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp.
==============================================================================*/

#ifndef HYBRIDBACKEND_COMMON_MURMUR3_CU_H_
#define HYBRIDBACKEND_COMMON_MURMUR3_CU_H_

inline __host__ __device__ uint32_t _rotl32(uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}

template <typename T, uint32_t seed = 0>
inline __host__ __device__ uint32_t murmur3_hash32(const T& input) {
  constexpr int len = sizeof(T);
  const uint8_t* const data = (const uint8_t*)&input;
  constexpr int nblocks = len / 4;
  uint32_t h1 = seed;
  constexpr uint32_t c1 = 0xcc9e2d51;
  constexpr uint32_t c2 = 0x1b873593;

  // body
  const uint32_t* const blocks = (const uint32_t*)(data + nblocks * 4);
  for (int i = -nblocks; i; i++) {
    uint32_t k1 = blocks[i];
    k1 *= c1;
    k1 = _rotl32(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = _rotl32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
  }

  // tail
  const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
  uint32_t k1 = 0;
  switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
    case 2:
      k1 ^= tail[1] << 8;
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = _rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  }

  // finalization
  h1 ^= len;
  h1 ^= h1 >> 16;
  h1 *= 0x85ebca6b;
  h1 ^= h1 >> 13;
  h1 *= 0xc2b2ae35;
  h1 ^= h1 >> 16;
  return h1;
}

#endif  // HYBRIDBACKEND_COMMON_MURMUR3_CU_H_
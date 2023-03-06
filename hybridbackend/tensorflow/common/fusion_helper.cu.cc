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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/version.h>

#include "hybridbackend/tensorflow/common/device_functions.h"
#include "hybridbackend/tensorflow/common/fusion_helper.cu.h"

namespace tensorflow {
namespace hybridbackend {
namespace functor {

template <typename T>
void CopyPtrsNFunctor<T>::operator()(OpKernelContext* ctx, int8* head_host,
                                     int8* head_device,
                                     std::vector<const Tensor*>* inputs,
                                     int num_columns) {
  T** head_host_ptr = reinterpret_cast<T**>(head_host);
  for (int i = 0; i < num_columns; ++i) {
    head_host_ptr[i] =
        const_cast<T*>((*inputs)[i]->flat_outer_dims<T>().data());
  }
  auto* stream = ctx->op_device_context()->stream();
  se::DeviceMemoryBase dst_ptr(head_device, num_columns * sizeof(T*));
  stream->ThenMemcpy(&dst_ptr, head_host, num_columns * sizeof(T*));
  stream->BlockHostUntilDone();
}

template <typename T>
void CopySizesNFunctor<T>::operator()(OpKernelContext* ctx, T* input_host,
                                      T* input_device, int num_columns) {
  auto* stream = ctx->op_device_context()->stream();
  se::DeviceMemoryBase dst_ptr(input_device, num_columns * sizeof(T));
  stream->ThenMemcpy(&dst_ptr, input_host, num_columns * sizeof(T));
  stream->BlockHostUntilDone();
}

#define DEFINE_COPY_PTRS(T) template struct CopyPtrsNFunctor<T>;
#define DEFINE_COPY_SIZES(T) template struct CopySizesNFunctor<T>;

#define TF_CALL_HELPER_TYPES(m) \
  TF_CALL_uint32(m) TF_CALL_uint64(m) TF_CALL_REAL_NUMBER_TYPES(m)

TF_CALL_HELPER_TYPES(DEFINE_COPY_PTRS);
TF_CALL_HELPER_TYPES(DEFINE_COPY_SIZES);

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

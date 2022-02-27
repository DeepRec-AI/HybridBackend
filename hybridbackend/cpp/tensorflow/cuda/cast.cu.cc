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

#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/version.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "hybridbackend/cpp/tensorflow/cuda/cast.h"
#include "hybridbackend/cpp/tensorflow/cuda/device_functions.h"
#include "hybridbackend/cpp/tensorflow/cuda/stream.h"

namespace tensorflow {
namespace hybridbackend {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
void CastToFloat16<T>::operator()(OpKernelContext* ctx, const Tensor* in,
                                  Tensor* out, cudaStream_t* comm_stream) {
  auto input = in->flat<T>();
  auto output = out->flat<Eigen::half>();
  const GPUDevice d = ctx->eigen_device<GPUDevice>();
  int64 thread_per_block, block_count;
  if (TF_PREDICT_TRUE(in->NumElements() > 0)) {
    GenerateCudaLaunchConfig(in->NumElements(), d, &thread_per_block,
                             &block_count);
    CastToFp16<T><<<block_count, thread_per_block, 0, *comm_stream>>>(
        input.data(), reinterpret_cast<__half*>(output.data()),
        in->NumElements());
  }
}

template <typename T>
void CastFromFloat16<T>::operator()(OpKernelContext* ctx, const Tensor* in,
                                    Tensor* out, cudaStream_t* comm_stream) {
  auto input = in->flat<Eigen::half>();
  auto output = out->flat<T>();
  const GPUDevice d = ctx->eigen_device<GPUDevice>();
  int64 thread_per_block, block_count;
  if (TF_PREDICT_TRUE(in->NumElements() > 0)) {
    GenerateCudaLaunchConfig(in->NumElements(), d, &thread_per_block,
                             &block_count);
    CastFromFp16<T><<<block_count, thread_per_block, 0, *comm_stream>>>(
        reinterpret_cast<const __half*>(input.data()), output.data(),
        in->NumElements());
  }
}

template struct CastToFloat16<float>;
template struct CastFromFloat16<float>;

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

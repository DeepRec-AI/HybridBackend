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
#ifndef HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_COMMON_CAST_FUNCTORS_H_
#define HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_COMMON_CAST_FUNCTORS_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_reference.h>
#include <tensorflow/core/public/version.h>

#if GOOGLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <tensorflow/core/common_runtime/gpu/gpu_event_mgr.h>
#include <tensorflow/stream_executor/stream_executor.h>
#endif

namespace tensorflow {

class OpKernelContext;

namespace hybridbackend {
namespace functor {

#if GOOGLE_CUDA
template <typename Tin, typename Tout>
struct Cast {
  void operator()(const Tensor& in, Tensor* out, OpKernelContext* ctx,
                  cudaStream_t* stream);
};

template <typename Tin, typename Tout>
struct CastN {
  void operator()(const std::vector<Tensor>& in, std::vector<Tensor>* out,
                  OpKernelContext* ctx, cudaStream_t* stream);
  void operator()(const std::vector<Tensor>& in, std::vector<Tensor*>* out,
                  OpKernelContext* ctx, cudaStream_t* stream);
  void operator()(const std::vector<Tensor*>& in, std::vector<Tensor*>* out,
                  OpKernelContext* ctx, cudaStream_t* stream);
};

#endif  // GOOGLE_CUDA

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW
#endif  // HYBRIDBACKEND_TENSORFLOW_DISTRIBUTE_COMMON_CAST_FUNCTORS_H_

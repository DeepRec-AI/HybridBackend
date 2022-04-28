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

__global__ void CastFp32ToFp16(const float* d_in, __half* d_out,
                               const size_t size) {
  for (size_t idx : CudaGridRangeX(size)) {
    d_out[idx] = __float2half(d_in[idx]);
  }
}

template <>
void Cast<float, Eigen::half>::operator()(const Tensor& in, Tensor* out,
                                          OpKernelContext* ctx,
                                          cudaStream_t* stream) {
  const size_t size = in.NumElements();
  if (TF_PREDICT_FALSE(size <= 0)) {
    return;
  }
  const float* d_input = in.flat<float>().data();
  __half* d_output = reinterpret_cast<__half*>(out->flat<Eigen::half>().data());
  CudaLaunch(CastFp32ToFp16, size, 0, ctx->eigen_device<GPUDevice>(), stream,
             d_input, d_output, size);
}

template struct Cast<float, Eigen::half>;

__global__ void CastFp16ToFp32(const __half* d_in, float* d_out,
                               const size_t size) {
  for (size_t idx : CudaGridRangeX(size)) {
    d_out[idx] = __half2float(d_in[idx]);
  }
}

template <>
void Cast<Eigen::half, float>::operator()(const Tensor& in, Tensor* out,
                                          OpKernelContext* ctx,
                                          cudaStream_t* stream) {
  const size_t size = in.NumElements();
  if (TF_PREDICT_FALSE(size <= 0)) {
    return;
  }
  const __half* d_input =
      reinterpret_cast<const __half*>(in.flat<Eigen::half>().data());
  float* d_output = out->flat<float>().data();
  CudaLaunch(CastFp16ToFp32, size, 0, ctx->eigen_device<GPUDevice>(), stream,
             d_input, d_output, size);
}

template struct Cast<Eigen::half, float>;

__global__ void GroupCastFp32ToFp16(const size_t total_max_in_size,
                                    const size_t max_in_size,
                                    const float** dd_in, __half** dd_out,
                                    const int32* d_size) {
  for (int32 idx : CudaGridRangeX(total_max_in_size)) {
    const int32 s = idx / max_in_size;
    const int32 sidx = idx % max_in_size;
    if (sidx < ldg(d_size + s)) {
      dd_out[s][sidx] = __float2half(dd_in[s][sidx]);
    }
  }
}

template <>
void GroupCast<float, Eigen::half>::operator()(const std::vector<Tensor>& in,
                                               std::vector<Tensor>* out,
                                               OpKernelContext* ctx,
                                               cudaStream_t* stream) {
  const int64 num_inputs = in.size();

  auto d = ctx->eigen_device<GPUDevice>();
  auto* ctx_stream = ctx->op_device_context()->stream();
  CudaStream ctx_cu_stream = CudaStream(ctx_stream);
  CudaStream cu_stream = CudaStream(stream);
  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_on_host(true);
  host_alloc_attrs.set_gpu_compatible(true);

  Tensor h_size_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}),
                                         &h_size_t, host_alloc_attrs));
  Tensor d_size_t;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}), &d_size_t));
  int32* h_size = h_size_t.flat<int32>().data();
  int32* d_size = d_size_t.flat<int32>().data();
  const int32 buffer_size =
      static_cast<int32>((sizeof(float*) + sizeof(Eigen::half*)) * num_inputs);

  Tensor h_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &h_buffer_t, host_alloc_attrs));
  Tensor d_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &d_buffer_t));
  int8* h_buffer = h_buffer_t.flat<int8>().data();
  int8* d_buffer = d_buffer_t.flat<int8>().data();

  const float** hd_in = reinterpret_cast<const float**>(h_buffer);
  const float** dd_in = reinterpret_cast<const float**>(d_buffer);
  __half** hd_out =
      reinterpret_cast<__half**>(h_buffer + sizeof(float*) * num_inputs);
  __half** dd_out =
      reinterpret_cast<__half**>(h_buffer + sizeof(float*) * num_inputs);

  size_t max_in_size = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    const size_t size = in[i].NumElements();
    h_size[i] = size;
    hd_in[i] = in[i].flat<float>().data();
    hd_out[i] =
        reinterpret_cast<__half*>(out->at(i).flat<Eigen::half>().data());
    if (size > max_in_size) {
      max_in_size = size;
    }
  }

  ctx_cu_stream.ThenCopyToDevice(d_size, h_size, sizeof(int32) * num_inputs);
  ctx_cu_stream.ThenCopyToDevice(d_buffer, h_buffer, buffer_size);
  cu_stream.ThenWaitFor(ctx_cu_stream);

  auto total_max_in_size = num_inputs * max_in_size;
  CudaLaunch(GroupCastFp32ToFp16, total_max_in_size, 0, d, stream,
             total_max_in_size, max_in_size, dd_in, dd_out, d_size);
}

template <>
void GroupCast<float, Eigen::half>::operator()(const std::vector<Tensor>& in,
                                               std::vector<Tensor*>* out,
                                               OpKernelContext* ctx,
                                               cudaStream_t* stream) {
  const int64 num_inputs = in.size();

  auto d = ctx->eigen_device<GPUDevice>();
  auto* ctx_stream = ctx->op_device_context()->stream();
  CudaStream ctx_cu_stream = CudaStream(ctx_stream);
  CudaStream cu_stream = CudaStream(stream);
  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_on_host(true);
  host_alloc_attrs.set_gpu_compatible(true);

  Tensor h_size_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}),
                                         &h_size_t, host_alloc_attrs));
  Tensor d_size_t;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}), &d_size_t));
  int32* h_size = h_size_t.flat<int32>().data();
  int32* d_size = d_size_t.flat<int32>().data();
  const int32 buffer_size =
      static_cast<int32>((sizeof(float*) + sizeof(Eigen::half*)) * num_inputs);

  Tensor h_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &h_buffer_t, host_alloc_attrs));
  Tensor d_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &d_buffer_t));
  int8* h_buffer = h_buffer_t.flat<int8>().data();
  int8* d_buffer = d_buffer_t.flat<int8>().data();

  const float** hd_in = reinterpret_cast<const float**>(h_buffer);
  const float** dd_in = reinterpret_cast<const float**>(d_buffer);
  __half** hd_out =
      reinterpret_cast<__half**>(h_buffer + sizeof(float*) * num_inputs);
  __half** dd_out =
      reinterpret_cast<__half**>(h_buffer + sizeof(float*) * num_inputs);

  size_t max_in_size = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    const size_t size = in[i].NumElements();
    h_size[i] = size;
    hd_in[i] = in[i].flat<float>().data();
    hd_out[i] =
        reinterpret_cast<__half*>(out->at(i)->flat<Eigen::half>().data());
    if (size > max_in_size) {
      max_in_size = size;
    }
  }

  ctx_cu_stream.ThenCopyToDevice(d_size, h_size, sizeof(int32) * num_inputs);
  ctx_cu_stream.ThenCopyToDevice(d_buffer, h_buffer, buffer_size);
  cu_stream.ThenWaitFor(ctx_cu_stream);

  auto total_max_in_size = num_inputs * max_in_size;
  CudaLaunch(GroupCastFp32ToFp16, total_max_in_size, 0, d, stream,
             total_max_in_size, max_in_size, dd_in, dd_out, d_size);
}

template <>
void GroupCast<float, Eigen::half>::operator()(const std::vector<Tensor*>& in,
                                               std::vector<Tensor*>* out,
                                               OpKernelContext* ctx,
                                               cudaStream_t* stream) {
  const int64 num_inputs = in.size();

  auto d = ctx->eigen_device<GPUDevice>();
  auto* ctx_stream = ctx->op_device_context()->stream();
  CudaStream ctx_cu_stream = CudaStream(ctx_stream);
  CudaStream cu_stream = CudaStream(stream);
  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_on_host(true);
  host_alloc_attrs.set_gpu_compatible(true);

  Tensor h_size_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}),
                                         &h_size_t, host_alloc_attrs));
  Tensor d_size_t;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}), &d_size_t));
  int32* h_size = h_size_t.flat<int32>().data();
  int32* d_size = d_size_t.flat<int32>().data();
  const int32 buffer_size =
      static_cast<int32>((sizeof(float*) + sizeof(Eigen::half*)) * num_inputs);

  Tensor h_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &h_buffer_t, host_alloc_attrs));
  Tensor d_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &d_buffer_t));
  int8* h_buffer = h_buffer_t.flat<int8>().data();
  int8* d_buffer = d_buffer_t.flat<int8>().data();

  const float** hd_in = reinterpret_cast<const float**>(h_buffer);
  const float** dd_in = reinterpret_cast<const float**>(d_buffer);
  __half** hd_out =
      reinterpret_cast<__half**>(h_buffer + sizeof(float*) * num_inputs);
  __half** dd_out =
      reinterpret_cast<__half**>(h_buffer + sizeof(float*) * num_inputs);

  size_t max_in_size = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    const size_t size = in[i]->NumElements();
    h_size[i] = size;
    hd_in[i] = in[i]->flat<float>().data();
    hd_out[i] =
        reinterpret_cast<__half*>(out->at(i)->flat<Eigen::half>().data());
    if (size > max_in_size) {
      max_in_size = size;
    }
  }

  ctx_cu_stream.ThenCopyToDevice(d_size, h_size, sizeof(int32) * num_inputs);
  ctx_cu_stream.ThenCopyToDevice(d_buffer, h_buffer, buffer_size);
  cu_stream.ThenWaitFor(ctx_cu_stream);

  auto total_max_in_size = num_inputs * max_in_size;
  CudaLaunch(GroupCastFp32ToFp16, total_max_in_size, 0, d, stream,
             total_max_in_size, max_in_size, dd_in, dd_out, d_size);
}

template struct GroupCast<float, Eigen::half>;

__global__ void GroupCastFp16ToFp32(const size_t total_max_in_size,
                                    const size_t max_in_size,
                                    const __half** dd_in, float** dd_out,
                                    const int32* d_size) {
  for (int32 idx : CudaGridRangeX(total_max_in_size)) {
    const int32 s = idx / max_in_size;
    const int32 sidx = idx % max_in_size;
    if (sidx < ldg(d_size + s)) {
      dd_out[s][sidx] = __half2float(dd_in[s][sidx]);
    }
  }
}

template <>
void GroupCast<Eigen::half, float>::operator()(const std::vector<Tensor>& in,
                                               std::vector<Tensor>* out,
                                               OpKernelContext* ctx,
                                               cudaStream_t* stream) {
  const int64 num_inputs = in.size();

  auto d = ctx->eigen_device<GPUDevice>();
  auto* ctx_stream = ctx->op_device_context()->stream();
  CudaStream ctx_cu_stream = CudaStream(ctx_stream);
  CudaStream cu_stream = CudaStream(stream);
  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_on_host(true);
  host_alloc_attrs.set_gpu_compatible(true);

  Tensor h_size_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}),
                                         &h_size_t, host_alloc_attrs));
  Tensor d_size_t;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}), &d_size_t));
  int32* h_size = h_size_t.flat<int32>().data();
  int32* d_size = d_size_t.flat<int32>().data();
  const int32 buffer_size =
      static_cast<int32>((sizeof(float*) + sizeof(Eigen::half*)) * num_inputs);

  Tensor h_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &h_buffer_t, host_alloc_attrs));
  Tensor d_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &d_buffer_t));
  int8* h_buffer = h_buffer_t.flat<int8>().data();
  int8* d_buffer = d_buffer_t.flat<int8>().data();

  const __half** hd_in = reinterpret_cast<const __half**>(h_buffer);
  const __half** dd_in = reinterpret_cast<const __half**>(d_buffer);
  float** hd_out =
      reinterpret_cast<float**>(h_buffer + sizeof(Eigen::half*) * num_inputs);
  float** dd_out =
      reinterpret_cast<float**>(d_buffer + sizeof(Eigen::half*) * num_inputs);

  size_t max_in_size = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    const size_t size = in[i].NumElements();
    h_size[i] = size;
    hd_in[i] =
        reinterpret_cast<const __half*>(in[i].flat<Eigen::half>().data());
    hd_out[i] = out->at(i).flat<float>().data();
    if (size > max_in_size) {
      max_in_size = size;
    }
  }

  ctx_cu_stream.ThenCopyToDevice(d_size, h_size, sizeof(int32) * num_inputs);
  ctx_cu_stream.ThenCopyToDevice(d_buffer, h_buffer, buffer_size);
  cu_stream.ThenWaitFor(ctx_cu_stream);

  auto total_max_in_size = num_inputs * max_in_size;
  CudaLaunch(GroupCastFp16ToFp32, total_max_in_size, 0, d, stream,
             total_max_in_size, max_in_size, dd_in, dd_out, d_size);
}

template <>
void GroupCast<Eigen::half, float>::operator()(const std::vector<Tensor>& in,
                                               std::vector<Tensor*>* out,
                                               OpKernelContext* ctx,
                                               cudaStream_t* stream) {
  const int64 num_inputs = in.size();

  auto d = ctx->eigen_device<GPUDevice>();
  auto* ctx_stream = ctx->op_device_context()->stream();
  CudaStream ctx_cu_stream = CudaStream(ctx_stream);
  CudaStream cu_stream = CudaStream(stream);
  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_on_host(true);
  host_alloc_attrs.set_gpu_compatible(true);

  Tensor h_size_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}),
                                         &h_size_t, host_alloc_attrs));
  Tensor d_size_t;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}), &d_size_t));
  int32* h_size = h_size_t.flat<int32>().data();
  int32* d_size = d_size_t.flat<int32>().data();
  const int32 buffer_size =
      static_cast<int32>((sizeof(float*) + sizeof(Eigen::half*)) * num_inputs);

  Tensor h_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &h_buffer_t, host_alloc_attrs));
  Tensor d_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &d_buffer_t));
  int8* h_buffer = h_buffer_t.flat<int8>().data();
  int8* d_buffer = d_buffer_t.flat<int8>().data();

  const __half** hd_in = reinterpret_cast<const __half**>(h_buffer);
  const __half** dd_in = reinterpret_cast<const __half**>(d_buffer);
  float** hd_out =
      reinterpret_cast<float**>(h_buffer + sizeof(Eigen::half*) * num_inputs);
  float** dd_out =
      reinterpret_cast<float**>(d_buffer + sizeof(Eigen::half*) * num_inputs);

  size_t max_in_size = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    const size_t size = in[i].NumElements();
    h_size[i] = size;
    hd_in[i] =
        reinterpret_cast<const __half*>(in[i].flat<Eigen::half>().data());
    hd_out[i] = out->at(i)->flat<float>().data();
    if (size > max_in_size) {
      max_in_size = size;
    }
  }

  ctx_cu_stream.ThenCopyToDevice(d_size, h_size, sizeof(int32) * num_inputs);
  ctx_cu_stream.ThenCopyToDevice(d_buffer, h_buffer, buffer_size);
  cu_stream.ThenWaitFor(ctx_cu_stream);

  auto total_max_in_size = num_inputs * max_in_size;
  CudaLaunch(GroupCastFp16ToFp32, total_max_in_size, 0, d, stream,
             total_max_in_size, max_in_size, dd_in, dd_out, d_size);
}

template <>
void GroupCast<Eigen::half, float>::operator()(const std::vector<Tensor*>& in,
                                               std::vector<Tensor*>* out,
                                               OpKernelContext* ctx,
                                               cudaStream_t* stream) {
  const int64 num_inputs = in.size();

  auto d = ctx->eigen_device<GPUDevice>();
  auto* ctx_stream = ctx->op_device_context()->stream();
  CudaStream ctx_cu_stream = CudaStream(ctx_stream);
  CudaStream cu_stream = CudaStream(stream);
  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_on_host(true);
  host_alloc_attrs.set_gpu_compatible(true);

  Tensor h_size_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}),
                                         &h_size_t, host_alloc_attrs));
  Tensor d_size_t;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_inputs}), &d_size_t));
  int32* h_size = h_size_t.flat<int32>().data();
  int32* d_size = d_size_t.flat<int32>().data();
  const int32 buffer_size =
      static_cast<int32>((sizeof(float*) + sizeof(Eigen::half*)) * num_inputs);

  Tensor h_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &h_buffer_t, host_alloc_attrs));
  Tensor d_buffer_t;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, TensorShape({buffer_size}),
                                         &d_buffer_t));
  int8* h_buffer = h_buffer_t.flat<int8>().data();
  int8* d_buffer = d_buffer_t.flat<int8>().data();

  const __half** hd_in = reinterpret_cast<const __half**>(h_buffer);
  const __half** dd_in = reinterpret_cast<const __half**>(d_buffer);
  float** hd_out =
      reinterpret_cast<float**>(h_buffer + sizeof(Eigen::half*) * num_inputs);
  float** dd_out =
      reinterpret_cast<float**>(d_buffer + sizeof(Eigen::half*) * num_inputs);

  size_t max_in_size = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    const size_t size = in[i]->NumElements();
    h_size[i] = size;
    hd_in[i] =
        reinterpret_cast<const __half*>(in[i]->flat<Eigen::half>().data());
    hd_out[i] = out->at(i)->flat<float>().data();
    if (size > max_in_size) {
      max_in_size = size;
    }
  }

  ctx_cu_stream.ThenCopyToDevice(d_size, h_size, sizeof(int32) * num_inputs);
  ctx_cu_stream.ThenCopyToDevice(d_buffer, h_buffer, buffer_size);
  cu_stream.ThenWaitFor(ctx_cu_stream);

  auto total_max_in_size = num_inputs * max_in_size;
  CudaLaunch(GroupCastFp16ToFp32, total_max_in_size, 0, d, stream,
             total_max_in_size, max_in_size, dd_in, dd_out, d_size);
}

template struct GroupCast<Eigen::half, float>;

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

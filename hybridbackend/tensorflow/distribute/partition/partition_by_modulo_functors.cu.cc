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

#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>

#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/version.h>

#include "hybridbackend/common/atomic.cu.h"
#include "hybridbackend/common/profiler.h"

#include "hybridbackend/tensorflow/common/device_functions.h"
#include "hybridbackend/tensorflow/distribute/partition/modulo_functors.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace hybridbackend {

namespace functor {

template <typename T>
__global__ void PartitionByModuloComputeSizes(const int32 num_partitions,
                                              const int32 input_size,
                                              const T* d_input, int32* d_sizes,
                                              int32* d_offsets) {
  for (int32 idx : CudaGridRangeX(input_size)) {
    const T v = d_input[idx];
    const T shard = (v % num_partitions + num_partitions) % num_partitions;
    d_offsets[idx] = atomicAdd(d_sizes + shard, 1);
  }
}

template <typename T>
__global__ void PartitionByModuloPopulate(const int32 num_partitions,
                                          const int32 input_size,
                                          const T* d_input,
                                          const int32* d_sizes,
                                          const int32* d_offsets, T* d_output,
                                          int32* d_indices) {
  extern __shared__ int32 outputs_offsets[];
  if (threadIdx.x == 0) {
    outputs_offsets[0] = 0;
    for (int i = 1; i < num_partitions; ++i) {
      outputs_offsets[i] = outputs_offsets[i - 1] + ldg(d_sizes + i - 1);
    }
  }
  __syncthreads();

  for (int32 idx : CudaGridRangeX(input_size)) {
    const T v = d_input[idx];
    const T shard = (v % num_partitions + num_partitions) % num_partitions;
    const int32 offset = d_offsets[idx] + outputs_offsets[shard];
    d_output[offset] = v;
    d_indices[idx] = offset;
  }
}

template <typename T>
struct PartitionByModulo<GPUDevice, T> {
  void operator()(const int32 num_partitions, const Tensor& input,
                  Tensor* output, Tensor* sizes, Tensor* indices,
                  OpKernelContext* ctx) {
    const int32 input_size = input.NumElements();
    const T* d_input = input.flat<T>().data();
    T* d_output = output->flat<T>().data();
    int32* d_sizes = sizes->flat<int32>().data();
    int32* d_indices = indices->flat<int32>().data();

    auto cu_stream = CudaStream(ctx->op_device_context()->stream());
    auto stream = *(cu_stream.get());
    auto d = ctx->eigen_device<GPUDevice>();
    cu_stream.ThenMemset(d_sizes, 0, num_partitions * sizeof(int32));

    if (TF_PREDICT_FALSE(input_size == 0)) {
      return;
    }

    auto* range = ::hybridbackend::ProfilerRange::forSynch("PartitionByModulo");
    Tensor offsets_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({input_size}),
                                           &offsets_t));
    int32* d_offsets = offsets_t.flat<int32>().data();

    CudaLaunch(PartitionByModuloComputeSizes<T>, input_size, 0, d, nullptr,
               num_partitions, input_size, d_input, d_sizes, d_offsets);
    CudaLaunch(PartitionByModuloPopulate<T>, input_size,
               num_partitions * sizeof(int32), d, nullptr, num_partitions,
               input_size, d_input, d_sizes, d_offsets, d_output, d_indices);
    ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        ctx->op_device_context()->stream(), [range]() { delete range; });
  }
};

template struct PartitionByModulo<GPUDevice, int32>;
template struct PartitionByModulo<GPUDevice, int64>;
template struct PartitionByModulo<GPUDevice, uint32>;
template struct PartitionByModulo<GPUDevice, uint64>;

template <typename T>
__global__ void PartitionByModuloNZeroSizes(const int32 total_num_partitions,
                                            const int32 num_partitions,
                                            int32** hd_outputs_sizes) {
  for (int32 idx : CudaGridRangeX(total_num_partitions)) {
    const int32 s = idx / num_partitions;
    const int32 sidx = idx % num_partitions;
    hd_outputs_sizes[s][sidx] = 0;
  }
}

template <typename T>
__global__ void PartitionByModuloNComputeSizes(
    const int32 total_max_inputs_size, const int32 max_inputs_size,
    const int32 num_partitions, const int32* d_inputs_size,
    const size_t* dd_inputs, size_t* dd_outputs_sizes,
    int32* d_output_segment_offsets) {
  for (int32 idx : CudaGridRangeX(total_max_inputs_size)) {
    const int32 s = idx / max_inputs_size;
    const int32 sidx = idx % max_inputs_size;
    if (sidx < ldg(d_inputs_size + s)) {
      const T* d_input = reinterpret_cast<const T*>(ldg(dd_inputs + s));
      const T v = d_input[sidx];
      const T shard = (v % num_partitions + num_partitions) % num_partitions;
      int32* d_sizes = reinterpret_cast<int32*>(ldg(dd_outputs_sizes + s));
      d_output_segment_offsets[idx] = atomicAdd(d_sizes + shard, 1);
    }
  }
}

__global__ void PartitionByModuloNComputeOffsets(const int32 num_inputs,
                                                 const int32 num_partitions,
                                                 size_t* dd_outputs_sizes,
                                                 int32* d_output_offsets) {
  for (int32 idx : CudaGridRangeX(num_inputs)) {
    int32* d_sizes = reinterpret_cast<int32*>(ldg(dd_outputs_sizes + idx));
    d_output_offsets[num_partitions * idx] = 0;
    for (int32 p = 1; p < num_partitions; ++p) {
      d_output_offsets[num_partitions * idx + p] =
          d_output_offsets[num_partitions * idx + p - 1] + ldg(d_sizes + p - 1);
    }
  }
}

template <typename T>
__global__ void PartitionByModuloNPopulate(
    const int32 total_max_inputs_size, const int32 max_inputs_size,
    const int32 num_partitions, const int32* d_inputs_size,
    const size_t* dd_inputs, size_t* dd_outputs, size_t* dd_outputs_indices,
    int32* d_output_segment_offsets, int32* d_output_offsets) {
  for (int32 idx : CudaGridRangeX(total_max_inputs_size)) {
    const int32 s = idx / max_inputs_size;
    const int32 sidx = idx % max_inputs_size;
    if (sidx < ldg(d_inputs_size + s)) {
      const T* d_input = reinterpret_cast<const T*>(ldg(dd_inputs + s));
      const T v = d_input[sidx];
      const T shard = (v % num_partitions + num_partitions) % num_partitions;
      const int32 soffset = d_output_segment_offsets[idx] +
                            ldg(d_output_offsets + num_partitions * s + shard);
      T* d_output = reinterpret_cast<T*>(ldg(dd_outputs + s));
      d_output[soffset] = v;
      int32* d_indices = reinterpret_cast<int32*>(ldg(dd_outputs_indices + s));
      d_indices[sidx] = soffset;
    }
  }
}

template <typename T>
struct PartitionByModuloN<GPUDevice, T> {
  void operator()(const int32 num_partitions, const std::vector<Tensor>& inputs,
                  std::vector<Tensor*>& outputs,
                  std::vector<Tensor*>& outputs_sizes,
                  std::vector<Tensor*>& outputs_indices, OpKernelContext* ctx) {
    auto* range =
        ::hybridbackend::ProfilerRange::forSynch("PartitionByModuloN");
    const int32 num_inputs = inputs.size();
    std::vector<int32> inputs_size;
    for (int i = 0; i < num_inputs; ++i) {
      inputs_size.push_back(inputs[i].NumElements());
    }
    int32 block_size = 0;
    int32 min_grid_size = 0;
    int32 grid_size = 0;

    auto cu_stream = CudaStream(ctx->op_device_context()->stream());
    auto d = ctx->eigen_device<GPUDevice>();

    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_on_host(true);
    host_alloc_attrs.set_gpu_compatible(true);

    int32 num_nonzero_inputs = 0;
    for (int32 i = 0; i < num_inputs; ++i) {
      if (TF_PREDICT_TRUE(inputs_size[i] > 0)) {
        num_nonzero_inputs++;
      }
    }

    Tensor hd_all_outputs_sizes_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DT_INT8,
                 TensorShape({num_inputs * static_cast<int32>(sizeof(int32*))}),
                 &hd_all_outputs_sizes_t, host_alloc_attrs));
    int32** hd_all_outputs_sizes =
        reinterpret_cast<int32**>(hd_all_outputs_sizes_t.flat<int8>().data());
    for (int32 i = 0; i < num_inputs; ++i) {
      hd_all_outputs_sizes[i] = outputs_sizes[i]->flat<int32>().data();
    }

    const int32 total_num_partitions = num_inputs * num_partitions;
    CudaLaunch(PartitionByModuloNZeroSizes<T>, total_num_partitions, 0, d,
               nullptr, total_num_partitions, num_partitions,
               hd_all_outputs_sizes);

    if (TF_PREDICT_FALSE(num_nonzero_inputs == 0)) {
      TensorReference ref_hd_all_outputs_sizes_t(hd_all_outputs_sizes_t);
      ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
          ctx->op_device_context()->stream(), [ref_hd_all_outputs_sizes_t]() {
            ref_hd_all_outputs_sizes_t.Unref();
          });
      return;
    }

    Tensor h_inputs_size_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_nonzero_inputs}),
                                &h_inputs_size_t, host_alloc_attrs));
    int32* h_inputs_size = h_inputs_size_t.flat<int32>().data();

    const int32 ptrs_bytes =
        num_nonzero_inputs * static_cast<int32>(sizeof(T*));
    Tensor hd_ptrs_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT8, TensorShape({4 * ptrs_bytes}),
                                      &hd_ptrs_t, host_alloc_attrs));
    int8* hd_ptrs = hd_ptrs_t.flat<int8>().data();
    const T** hd_inputs = reinterpret_cast<const T**>(hd_ptrs);
    T** hd_outputs = reinterpret_cast<T**>(hd_ptrs + ptrs_bytes);
    int32** hd_outputs_sizes =
        reinterpret_cast<int32**>(hd_ptrs + 2 * ptrs_bytes);
    int32** hd_outputs_indices =
        reinterpret_cast<int32**>(hd_ptrs + 3 * ptrs_bytes);
    for (int32 i = 0, j = 0; i < num_inputs; ++i) {
      if (TF_PREDICT_TRUE(inputs_size[i] > 0)) {
        h_inputs_size[j] = inputs_size[i];
        hd_inputs[j] = inputs[i].flat<T>().data();
        hd_outputs[j] = outputs[i]->flat<T>().data();
        hd_outputs_sizes[j] = outputs_sizes[i]->flat<int32>().data();
        hd_outputs_indices[j] = outputs_indices[i]->flat<int32>().data();
        ++j;
      }
    }
    int32 max_inputs_size = h_inputs_size[0];
    for (int32 i = 1; i < num_nonzero_inputs; ++i) {
      if (max_inputs_size < h_inputs_size[i]) {
        max_inputs_size = h_inputs_size[i];
      }
    }
    const int32 total_max_inputs_size = max_inputs_size * num_nonzero_inputs;

    Tensor d_inputs_size_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_INT32, TensorShape({num_nonzero_inputs}),
                                &d_inputs_size_t));
    int32* d_inputs_size = d_inputs_size_t.flat<int32>().data();
    se::DeviceMemoryBase d_inputs_size_ptr(d_inputs_size,
                                           d_inputs_size_t.TotalBytes());
    ctx->op_device_context()->stream()->ThenMemcpy(
        &d_inputs_size_ptr, h_inputs_size, d_inputs_size_t.TotalBytes());

    Tensor dd_ptrs_t;
    OP_REQUIRES_OK(
        ctx,
        ctx->allocate_temp(DT_INT8, TensorShape({4 * ptrs_bytes}), &dd_ptrs_t));
    int8* dd_ptrs = dd_ptrs_t.flat<int8>().data();
    se::DeviceMemoryBase dd_ptrs_ptr(dd_ptrs, dd_ptrs_t.TotalBytes());
    ctx->op_device_context()->stream()->ThenMemcpy(&dd_ptrs_ptr, hd_ptrs,
                                                   dd_ptrs_t.TotalBytes());
    const size_t* dd_inputs = reinterpret_cast<const size_t*>(dd_ptrs);
    size_t* dd_outputs = reinterpret_cast<size_t*>(dd_ptrs + ptrs_bytes);
    size_t* dd_outputs_sizes =
        reinterpret_cast<size_t*>(dd_ptrs + 2 * ptrs_bytes);
    size_t* dd_outputs_indices =
        reinterpret_cast<size_t*>(dd_ptrs + 3 * ptrs_bytes);

    Tensor d_offsets_buffer_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DT_INT32,
                            TensorShape({total_max_inputs_size +
                                         num_nonzero_inputs * num_partitions}),
                            &d_offsets_buffer_t));
    int32* d_output_segment_offsets = d_offsets_buffer_t.flat<int32>().data();
    int32* d_output_offsets =
        d_offsets_buffer_t.flat<int32>().data() + total_max_inputs_size;

    CudaLaunch(PartitionByModuloNComputeSizes<T>, total_max_inputs_size, 0, d,
               nullptr, total_max_inputs_size, max_inputs_size, num_partitions,
               d_inputs_size, dd_inputs, dd_outputs_sizes,
               d_output_segment_offsets);
    CudaLaunch(PartitionByModuloNComputeOffsets, num_nonzero_inputs, 0, d,
               nullptr, num_nonzero_inputs, num_partitions, dd_outputs_sizes,
               d_output_offsets);
    CudaLaunch(PartitionByModuloNPopulate<T>, total_max_inputs_size, 0, d,
               nullptr, total_max_inputs_size, max_inputs_size, num_partitions,
               d_inputs_size, dd_inputs, dd_outputs, dd_outputs_indices,
               d_output_segment_offsets, d_output_offsets);

    TensorReference ref_hd_all_outputs_sizes_t(hd_all_outputs_sizes_t);
    TensorReference ref_h_inputs_size_t(h_inputs_size_t);
    TensorReference ref_d_inputs_size_t(d_inputs_size_t);
    TensorReference ref_hd_ptrs_t(hd_ptrs_t);
    TensorReference ref_dd_ptrs_t(dd_ptrs_t);
    TensorReference ref_d_offsets_buffer_t(d_offsets_buffer_t);
    ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        ctx->op_device_context()->stream(),
        [range, ref_hd_all_outputs_sizes_t, ref_h_inputs_size_t,
         ref_d_inputs_size_t, ref_hd_ptrs_t, ref_dd_ptrs_t,
         ref_d_offsets_buffer_t]() {
          delete range;
          ref_hd_all_outputs_sizes_t.Unref();
          ref_h_inputs_size_t.Unref();
          ref_d_inputs_size_t.Unref();
          ref_hd_ptrs_t.Unref();
          ref_dd_ptrs_t.Unref();
          ref_d_offsets_buffer_t.Unref();
        });
  }
};

template struct PartitionByModuloN<GPUDevice, int32>;
template struct PartitionByModuloN<GPUDevice, int64>;
template struct PartitionByModuloN<GPUDevice, uint32>;
template struct PartitionByModuloN<GPUDevice, uint64>;

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

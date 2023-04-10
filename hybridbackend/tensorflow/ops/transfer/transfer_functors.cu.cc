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

#include "hybridbackend/common/atomic.cu.h"
#include "hybridbackend/tensorflow/common/device_functions.h"
#include "hybridbackend/tensorflow/common/fusion_helper.cu.h"
#include "hybridbackend/tensorflow/ops/transfer/functors.h"

namespace tensorflow {
namespace hybridbackend {
using GPUDevice = Eigen::GpuDevice;

template <typename T>
__global__ void MemcpyH2DNKernel(int64 h_index_x, int64 h_index_y,
                                 int64* h_pinned_output_sizes,
                                 int64* h_pinned_input_sizes, T** output,
                                 T** input) {
  const int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64 g_idx = idx / h_index_y;
  int64 g_offset = idx % h_index_y;
  if (g_idx < h_index_x && g_offset < h_pinned_output_sizes[g_idx]) {
    output[g_idx][g_offset] = static_cast<T>(ldg(input[g_idx] + g_offset));
  }
}

namespace functor {

template <typename T>
TransferH2DNFunctor<T>::TransferH2DNFunctor(const OpInputList& inputs,
                                            OpOutputList& outputs,
                                            OpKernelContext* ctx)
    : h_pinned_buffer_tensor_(nullptr),
      d_pinned_buffer_tensor_(nullptr),
      h_unpinned_fusion_buffer_tensor_(nullptr),
      pinned_input_bytes_(0),
      unpinned_input_bytes_(0),
      num_pinned_inputs_(0),
      num_unpinned_inputs_(0) {
  std::vector<int32> pinned_input_indices;
  std::vector<int32> all_unpinned_input_indices;
  for (int32 idx = 0; idx < inputs.size(); ++idx) {
    if (inputs[idx].NumElements() == 0) {
      continue;
    }
    unsigned int host_flags = 0;
    auto rc = cudaHostGetFlags(
        &host_flags, const_cast<char*>(inputs[idx].tensor_data().data()));
    if (rc == cudaSuccess) {
      pinned_input_indices.push_back(idx);
      pinned_input_bytes_ += inputs[idx].TotalBytes();
    } else {
      all_unpinned_input_indices.push_back(idx);
      unpinned_input_bytes_ += inputs[idx].TotalBytes();
    }
  }

  std::vector<int32> unpinned_input_indices;
  std::vector<int32> unpinned_fusion_input_indices;
  static const int64 kMinMemcpyBytes = 1 << 15;  // 32KB
  int64 total_unpinned_fusion_size = 0;
  int64 unpinned_to_pinned_start = pinned_input_indices.size();
  for (int32 idx : all_unpinned_input_indices) {
    if (inputs[idx].TotalBytes() < kMinMemcpyBytes) {
      total_unpinned_fusion_size += inputs[idx].NumElements();
      unpinned_fusion_input_indices.push_back(idx);
      pinned_input_indices.push_back(idx);
    } else {
      unpinned_input_indices.push_back(idx);
    }
  }

  AllocatorAttributes host_alloc_attr;
  host_alloc_attr.set_on_host(true);
  host_alloc_attr.set_gpu_compatible(true);

  if (unpinned_fusion_input_indices.size() > 0) {
    h_unpinned_fusion_buffer_tensor_ = new Tensor;
    ctx->allocate_temp(DataTypeToEnum<T>::value,
                       TensorShape{total_unpinned_fusion_size},
                       h_unpinned_fusion_buffer_tensor_, host_alloc_attr);
    T* h_unpinned_fusion_buffer =
        h_unpinned_fusion_buffer_tensor_->flat<T>().data();
    size_t offset = 0;
    for (int32 idx : unpinned_fusion_input_indices) {
      unpinned_fusion_outputs_.push_back(h_unpinned_fusion_buffer + offset);
      unpinned_fusion_inputs_.push_back(inputs[idx].flat<T>().data());
      unpinned_fusion_bytes_.push_back(inputs[idx].TotalBytes());
      offset += inputs[idx].NumElements();
    }
  }

  num_pinned_inputs_ = pinned_input_indices.size();
  num_unpinned_inputs_ = unpinned_input_indices.size();

  for (int32 idx : unpinned_input_indices) {
    unpinned_outputs_.push_back(
        const_cast<char*>(outputs[idx]->tensor_data().data()));
    unpinned_inputs_.push_back(inputs[idx].tensor_data().data());
    unpinned_bytes_.push_back(inputs[idx].TotalBytes());
  }

  if (num_pinned_inputs_ > 0) {
    int64 pinned_ptrs_bytes = num_pinned_inputs_ * sizeof(T*);
    int64 pinned_sizes_bytes = num_pinned_inputs_ * sizeof(int64);
    pinned_buffer_bytes_ = pinned_ptrs_bytes * 2 + pinned_sizes_bytes * 2;

    // Pinned-Memory Buffer Structure:
    // input_raw_ptrs (pinned_ptrs_bytes)
    // output_raw_ptrs (pinned_ptrs_bytes)
    // input_sizes (pinned_sizes_bytes)
    // output_sizes (pinned_sizes_bytes)

    // Note: All element has at least 64bit width, which means segment address
    // can be accessed in CUDA kernel.

    h_pinned_buffer_tensor_ = new Tensor;
    ctx->allocate_temp(DT_INT8, TensorShape{pinned_buffer_bytes_},
                       h_pinned_buffer_tensor_, host_alloc_attr);
    d_pinned_buffer_tensor_ = new Tensor;
    ctx->allocate_temp(DT_INT8, TensorShape{pinned_buffer_bytes_},
                       d_pinned_buffer_tensor_);

    h_pinned_buffer_ = h_pinned_buffer_tensor_->flat<int8>().data();
    int8* h_pinned_input_raw_ptrs = h_pinned_buffer_;
    T** h_pinned_input_ptrs = reinterpret_cast<T**>(h_pinned_input_raw_ptrs);
    int8* h_pinned_output_raw_ptrs = h_pinned_buffer_ + pinned_ptrs_bytes;
    T** h_pinned_output_ptrs = reinterpret_cast<T**>(h_pinned_output_raw_ptrs);
    int64* h_pinned_input_sizes =
        reinterpret_cast<int64*>(h_pinned_buffer_ + pinned_ptrs_bytes * 2);
    int64* h_pinned_output_sizes = reinterpret_cast<int64*>(
        h_pinned_buffer_ + pinned_ptrs_bytes * 2 + pinned_sizes_bytes);
    for (int i = 0; i < unpinned_to_pinned_start; ++i) {
      h_pinned_input_ptrs[i] =
          const_cast<T*>(inputs[pinned_input_indices[i]].flat<T>().data());
      h_pinned_output_ptrs[i] =
          const_cast<T*>(outputs[pinned_input_indices[i]]->flat<T>().data());
      h_pinned_input_sizes[i] = inputs[pinned_input_indices[i]].NumElements();
      h_pinned_output_sizes[i] =
          outputs[pinned_input_indices[i]]->NumElements();
    }
    for (int i = unpinned_to_pinned_start; i < num_pinned_inputs_; ++i) {
      h_pinned_input_ptrs[i] =
          unpinned_fusion_outputs_[i - unpinned_to_pinned_start];
      h_pinned_output_ptrs[i] =
          const_cast<T*>(outputs[pinned_input_indices[i]]->flat<T>().data());
      h_pinned_input_sizes[i] = inputs[pinned_input_indices[i]].NumElements();
      h_pinned_output_sizes[i] =
          outputs[pinned_input_indices[i]]->NumElements();
    }

    max_pinned_output_size_ = *std::max_element(
        h_pinned_output_sizes, h_pinned_output_sizes + num_pinned_inputs_);

    d_pinned_buffer_ = d_pinned_buffer_tensor_->flat<int8>().data();
    d_pinned_input_raw_ptrs_ = d_pinned_buffer_;
    d_pinned_output_raw_ptrs_ = d_pinned_buffer_ + pinned_ptrs_bytes;
    d_pinned_input_sizes_ =
        reinterpret_cast<int64*>(d_pinned_buffer_ + pinned_ptrs_bytes * 2);
    d_pinned_output_sizes_ = reinterpret_cast<int64*>(
        d_pinned_buffer_ + pinned_ptrs_bytes * 2 + pinned_sizes_bytes);

    auto d = ctx->eigen_device<GPUDevice>();
    pinned_copy_block_size_ = d.maxGpuThreadsPerBlock();
  }
}

template <typename T>
TransferH2DNFunctor<T>::~TransferH2DNFunctor() {
  delete h_pinned_buffer_tensor_;
  delete d_pinned_buffer_tensor_;
  delete h_unpinned_fusion_buffer_tensor_;
}

template <typename T>
Status TransferH2DNFunctor<T>::Copy(cudaStream_t* stream) {
  for (int32 idx = 0; idx < unpinned_bytes_.size(); ++idx) {
    TF_RETURN_IF_ERROR(CudaErrorToStatus(cudaMemcpyAsync(
        unpinned_outputs_[idx], unpinned_inputs_[idx], unpinned_bytes_[idx],
        cudaMemcpyHostToDevice, *stream)));
  }

  if (num_pinned_inputs_ == 0) {
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(CudaErrorToStatus(
      cudaMemcpyAsync(d_pinned_buffer_, h_pinned_buffer_, pinned_buffer_bytes_,
                      cudaMemcpyHostToDevice, *stream)));

  for (int32 i = 0; i < unpinned_fusion_bytes_.size(); ++i) {
    memcpy(unpinned_fusion_outputs_[i], unpinned_fusion_inputs_[i],
           unpinned_fusion_bytes_[i]);
  }

  int64 total_size = max_pinned_output_size_ * num_pinned_inputs_;
  int grid_size = ((total_size - 1) / pinned_copy_block_size_) + 1;
  return CudaLaunchKernelInternal(
      MemcpyH2DNKernel<T>, grid_size, pinned_copy_block_size_, 0, *stream,
      num_pinned_inputs_, max_pinned_output_size_, d_pinned_output_sizes_,
      d_pinned_input_sizes_, reinterpret_cast<T**>(d_pinned_output_raw_ptrs_),
      reinterpret_cast<T**>(d_pinned_input_raw_ptrs_));
}

#define REGISTER_TRANSFER_H2D_N_FUNCTOR(T) \
  template struct TransferH2DNFunctor<T>;

TF_CALL_TRANSFER_TYPES(REGISTER_TRANSFER_H2D_N_FUNCTOR);

}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

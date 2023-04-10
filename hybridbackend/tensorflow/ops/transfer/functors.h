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

#ifndef HYBRIDBACKEND_TENSORFLOW_OPS_TRANSFER_FUNCTORS_H_
#define HYBRIDBACKEND_TENSORFLOW_OPS_TRANSFER_FUNCTORS_H_

#if HYBRIDBACKEND_TENSORFLOW

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_reference.h>
#include <tensorflow/core/public/version.h>

namespace tensorflow {

class OpKernelContext;

namespace hybridbackend {
namespace functor {

#define TF_CALL_TRANSFER_TYPES(m)                                         \
  TF_CALL_int8(m) TF_CALL_uint8(m) TF_CALL_int32(m) TF_CALL_uint32(m)     \
      TF_CALL_int64(m) TF_CALL_uint64(m) TF_CALL_half(m) TF_CALL_float(m) \
          TF_CALL_double(m)
#define TF_OP_TRANSFER_DTYPE_LIST \
  "int8, uint8, int32, uint32, int64, uint64, half, float, double"

#if GOOGLE_CUDA

template <typename T>
struct TransferH2DNFunctor {
 public:
  TransferH2DNFunctor(const OpInputList& inputs, OpOutputList& outputs,
                      OpKernelContext* ctx);
  virtual ~TransferH2DNFunctor();

  int64 num_pinned_inputs() const { return num_pinned_inputs_; }
  int64 num_unpinned_inputs() const { return num_unpinned_inputs_; }

  int64 pinned_input_bytes() const { return pinned_input_bytes_; }
  int64 unpinned_input_bytes() const { return unpinned_input_bytes_; }

  Status Copy(cudaStream_t* stream);

 private:
  int64 num_unpinned_inputs_;
  int64 unpinned_input_bytes_;
  std::vector<char*> unpinned_outputs_;
  std::vector<const void*> unpinned_inputs_;
  std::vector<size_t> unpinned_bytes_;
  Tensor* h_unpinned_fusion_buffer_tensor_;
  std::vector<T*> unpinned_fusion_outputs_;
  std::vector<const T*> unpinned_fusion_inputs_;
  std::vector<size_t> unpinned_fusion_bytes_;

  int64 num_pinned_inputs_;
  int64 pinned_input_bytes_;
  int64 pinned_buffer_bytes_;
  Tensor* h_pinned_buffer_tensor_;
  Tensor* d_pinned_buffer_tensor_;
  int8* h_pinned_buffer_;
  int8* d_pinned_buffer_;
  int8* d_pinned_input_raw_ptrs_;
  int8* d_pinned_output_raw_ptrs_;
  int64* d_pinned_input_sizes_;
  int64* d_pinned_output_sizes_;
  int64 max_pinned_output_size_;
  int pinned_copy_block_size_;
};

#endif  // GOOGLE_CUDA
}  // namespace functor
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW
#endif  // HYBRIDBACKEND_TENSORFLOW_OPS_TRANSFER_FUNCTORS_H_

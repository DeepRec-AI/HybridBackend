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

#include <bitset>

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.pb.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/tensorflow/distribute/partition/dual_modulo_functors.h"

#if GOOGLE_CUDA
#include "hybridbackend/tensorflow/common/host_functions.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
#if GOOGLE_CUDA
using GPUDevice = Eigen::GpuDevice;
#endif  // GOOGLE_CUDA

namespace hybridbackend {

REGISTER_OP("HbPartitionByDualModuloStageOne")
    .Output("output: T")
    .Output("sizes: int32")
    .Output("indices: int32")
    .Input("input: T")
    .Attr("T: {int32, int64, uint32, uint64}")
    .Attr("num_partitions: int >= 1")
    .Attr("modulus: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      int num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("num_partitions", &num_partitions));
      c->set_output(1, c->Vector(num_partitions));
      c->set_output(2, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Shuffle inputs into partitions.

output: Shuffling result with same shape of input.
sizes: Partition sizes in output.
indices: Indices for gathering output back to input.
input: Input vector.
num_partitions: Number of partitions.
modulus: modulus to calculate partition id.
)doc");

template <typename Device, typename T, typename Stage>
class PartitionByDualModuloOp : public OpKernel {
 public:
  PartitionByDualModuloOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("modulus", &modulus_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input.shape()),
                errors::InvalidArgument(
                    "partition_by_dual_modulo expects a 1D vector."));

    const int32 input_size = input.NumElements();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {input_size}, &output));

    Tensor* sizes = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {num_partitions_}, &sizes));

    Tensor* indices = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {input_size}, &indices));

    functor::PartitionByDualModulo<Device, T, Stage>()(
        num_partitions_, modulus_, input, output, sizes, indices, ctx);
  }

 private:
  int32 num_partitions_;
  int32 modulus_;
};

#define TF_CALL_SHUFFLE_PARTITION_TYPES(m) \
  TF_CALL_int32(m) TF_CALL_uint32(m) TF_CALL_int64(m) TF_CALL_uint64(m)

#define REGISTER_SHUFFLE_PARTITION_KERNEL(TYPE) \
  REGISTER_KERNEL_BUILDER(                      \
      Name("HbPartitionByDualModuloStageOne")   \
          .Device(DEVICE_CPU)                   \
          .TypeConstraint<TYPE>("T"),           \
      PartitionByDualModuloOp<CPUDevice, TYPE,  \
                              functor::ComputeShardAtStageOne>)
TF_CALL_SHUFFLE_PARTITION_TYPES(REGISTER_SHUFFLE_PARTITION_KERNEL);
#undef REGISTER_SHUFFLE_PARTITION_KERNEL

#if GOOGLE_CUDA
#define REGISTER_SHUFFLE_PARTITION_KERNEL(TYPE) \
  REGISTER_KERNEL_BUILDER(                      \
      Name("HbPartitionByDualModuloStageOne")   \
          .Device(DEVICE_GPU)                   \
          .TypeConstraint<TYPE>("T"),           \
      PartitionByDualModuloOp<GPUDevice, TYPE,  \
                              functor::ComputeShardOnGpuAtStageOne>)
TF_CALL_SHUFFLE_PARTITION_TYPES(REGISTER_SHUFFLE_PARTITION_KERNEL);
#undef REGISTER_SHUFFLE_PARTITION_KERNEL
#endif  // GOOGLE_CUDA

REGISTER_OP("HbPartitionByDualModuloStageTwo")
    .Output("output: T")
    .Output("sizes: int32")
    .Output("indices: int32")
    .Input("input: T")
    .Attr("T: {int32, int64, uint32, uint64}")
    .Attr("num_partitions: int >= 1")
    .Attr("modulus: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      int num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("num_partitions", &num_partitions));
      c->set_output(1, c->Vector(num_partitions));
      c->set_output(2, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Shuffle inputs into partitions.

output: Shuffling result with same shape of input.
sizes: Partition sizes in output.
indices: Indices for gathering output back to input.
input: Input vector.
num_partitions: Number of partitions.
modulus: modulus to calculate partition id.
)doc");

#define TF_CALL_SHUFFLE_PARTITION_TYPES(m) \
  TF_CALL_int32(m) TF_CALL_uint32(m) TF_CALL_int64(m) TF_CALL_uint64(m)

#define REGISTER_SHUFFLE_PARTITION_KERNEL(TYPE) \
  REGISTER_KERNEL_BUILDER(                      \
      Name("HbPartitionByDualModuloStageTwo")   \
          .Device(DEVICE_CPU)                   \
          .TypeConstraint<TYPE>("T"),           \
      PartitionByDualModuloOp<CPUDevice, TYPE,  \
                              functor::ComputeShardAtStageTwo>)
TF_CALL_SHUFFLE_PARTITION_TYPES(REGISTER_SHUFFLE_PARTITION_KERNEL);
#undef REGISTER_SHUFFLE_PARTITION_KERNEL

#if GOOGLE_CUDA
#define REGISTER_SHUFFLE_PARTITION_KERNEL(TYPE) \
  REGISTER_KERNEL_BUILDER(                      \
      Name("HbPartitionByDualModuloStageTwo")   \
          .Device(DEVICE_GPU)                   \
          .TypeConstraint<TYPE>("T"),           \
      PartitionByDualModuloOp<GPUDevice, TYPE,  \
                              functor::ComputeShardOnGpuAtStageTwo>)
TF_CALL_SHUFFLE_PARTITION_TYPES(REGISTER_SHUFFLE_PARTITION_KERNEL);
#undef REGISTER_SHUFFLE_PARTITION_KERNEL
#endif  // GOOGLE_CUDA

REGISTER_OP("HbPartitionByDualModuloStageOneN")
    .Output("outputs: N * T")
    .Output("outputs_sizes: N * int32")
    .Output("outputs_indices: N * int32")
    .Input("inputs: N * T")
    .Attr("N: int >= 1 = 1")
    .Attr("T: {int32, int64, uint32, uint64}")
    .Attr("num_partitions: int >= 1")
    .Attr("modulus: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 num_inputs;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_inputs));
      int num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("num_partitions", &num_partitions));
      for (int64 i = 0; i < num_inputs; ++i) {
        c->set_output(i, c->input(i));
        c->set_output(num_inputs + i, c->Vector(num_partitions));
        c->set_output(2 * num_inputs + i, c->input(i));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Shuffle multiple inputs into partitions.

outputs: Shuffling results with same shape of inputs.
outputs_sizes: Partition sizes in outputs.
outputs_indices: Indices for gathering outputs back to inputs.
inputs: Input vectors.
num_partitions: Number of partitions.
modulus: modulus to calculate partition id.
)doc");

template <typename Device, typename T, typename Stage>
class PartitionByDualModuloNOp : public OpKernel {
 public:
  PartitionByDualModuloNOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_inputs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("modulus", &modulus_));
  }

  void Compute(OpKernelContext* ctx) override {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));

    std::vector<Tensor> inputs_vec;
    std::vector<Tensor*> outputs_vec;
    std::vector<Tensor*> sizes_vec;
    std::vector<Tensor*> indices_vec;

    for (int i = 0; i < num_inputs_; ++i) {
      const Tensor& input = inputs[i];
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsVector(input.shape()),
          errors::InvalidArgument(
              "partition_by_dual_modulo_n expects 1D vector for input ", i));
      const int32 input_size = input.NumElements();
      inputs_vec.push_back(input);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {input_size}, &output));
      outputs_vec.push_back(output);
      Tensor* sizes = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(num_inputs_ + i,
                                               {num_partitions_}, &sizes));
      sizes_vec.push_back(sizes);
      Tensor* indices = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(2 * num_inputs_ + i,
                                               {input_size}, &indices));
      indices_vec.push_back(indices);
    }

    functor::PartitionByDualModuloN<Device, T, Stage>()(
        num_partitions_, modulus_, inputs_vec, outputs_vec, sizes_vec,
        indices_vec, ctx);
  }

 private:
  int32 num_inputs_;
  int32 num_partitions_;
  int32 modulus_;
};

#if GOOGLE_CUDA
#define REGISTER_SHUFFLE_PARTITION_KERNEL(TYPE) \
  REGISTER_KERNEL_BUILDER(                      \
      Name("HbPartitionByDualModuloStageOneN")  \
          .Device(DEVICE_GPU)                   \
          .TypeConstraint<TYPE>("T"),           \
      PartitionByDualModuloNOp<GPUDevice, TYPE, \
                               functor::ComputeShardOnGpuAtStageOne>)
TF_CALL_SHUFFLE_PARTITION_TYPES(REGISTER_SHUFFLE_PARTITION_KERNEL);
#undef REGISTER_SHUFFLE_PARTITION_KERNEL
#endif  // GOOGLE_CUDA

REGISTER_OP("HbPartitionByDualModuloStageTwoN")
    .Output("outputs: N * T")
    .Output("outputs_sizes: N * int32")
    .Output("outputs_indices: N * int32")
    .Input("inputs: N * T")
    .Attr("N: int >= 1 = 1")
    .Attr("T: {int32, int64, uint32, uint64}")
    .Attr("num_partitions: int >= 1")
    .Attr("modulus: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 num_inputs;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_inputs));
      int num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("num_partitions", &num_partitions));
      for (int64 i = 0; i < num_inputs; ++i) {
        c->set_output(i, c->input(i));
        c->set_output(num_inputs + i, c->Vector(num_partitions));
        c->set_output(2 * num_inputs + i, c->input(i));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Shuffle multiple inputs into partitions.

outputs: Shuffling results with same shape of inputs.
outputs_sizes: Partition sizes in outputs.
outputs_indices: Indices for gathering outputs back to inputs.
inputs: Input vectors.
num_partitions: Number of partitions.
modulus: modulus to calculate partition id.
)doc");

#if GOOGLE_CUDA
#define REGISTER_SHUFFLE_PARTITION_KERNEL(TYPE) \
  REGISTER_KERNEL_BUILDER(                      \
      Name("HbPartitionByDualModuloStageTwoN")  \
          .Device(DEVICE_GPU)                   \
          .TypeConstraint<TYPE>("T"),           \
      PartitionByDualModuloNOp<GPUDevice, TYPE, \
                               functor::ComputeShardOnGpuAtStageTwo>)
TF_CALL_SHUFFLE_PARTITION_TYPES(REGISTER_SHUFFLE_PARTITION_KERNEL);
#undef REGISTER_SHUFFLE_PARTITION_KERNEL
#endif  // GOOGLE_CUDA
}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

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

#include <bitset>

#include <tensorflow/core/common_runtime/optimization_registry.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor.pb.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/util/device_name_utils.h>

#include "hybridbackend/common/env.h"
#include "hybridbackend/common/profiler.h"
#include "hybridbackend/tensorflow/common/host_functions.h"
#include "hybridbackend/tensorflow/embedding/lookup_functors.h"

namespace tensorflow {
namespace hybridbackend {

REGISTER_OP("HbLookup")
    .Output("hit_keys_indices: Tindices")
    .Output("hit_cache_indices: T")
    .Output("miss_keys_indices: Tindices")
    .Output("miss_keys: T")
    .Input("keys_cache: T")
    .Input("keys: T")
    .Attr("T: type")
    .Attr("Tindices: {int32}")
    .Attr("cache_slab_size: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0,
                    c->Vector(shape_inference::InferenceContext::kUnknownDim));
      c->set_output(1,
                    c->Vector(shape_inference::InferenceContext::kUnknownDim));
      c->set_output(2,
                    c->Vector(shape_inference::InferenceContext::kUnknownDim));
      c->set_output(3,
                    c->Vector(shape_inference::InferenceContext::kUnknownDim));
      return Status::OK();
    });

using GPUDevice = Eigen::GpuDevice;

template <typename T>
class LookupOp : public OpKernel {
 public:
  LookupOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cache_slab_size", &cache_slab_size_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto* stream = ctx->op_device_context()->stream();

    const Tensor& keys_cache_tensor = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(keys_cache_tensor.shape()),
                errors::InvalidArgument("keys_cache expects a 1D vector."));
    const T* d_keys_cache = keys_cache_tensor.flat<T>().data();
    const T keys_cache_slab_count =
        static_cast<T>(keys_cache_tensor.NumElements() / cache_slab_size_);

    const Tensor& keys_tensor = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(keys_tensor.shape()),
                errors::InvalidArgument("keys expects a 1D vector."));
    const T* d_keys = keys_tensor.flat<T>().data();
    const int32 key_count = keys_tensor.NumElements();

    if (TF_PREDICT_FALSE(key_count == 0)) {
      Tensor* hit_keys_indices_ptr = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {0}, &hit_keys_indices_ptr));
      Tensor* hit_cache_indices_ptr = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {0}, &hit_cache_indices_ptr));
      Tensor* miss_keys_indices_ptr = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {0}, &miss_keys_indices_ptr));
      Tensor* miss_keys_ptr = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(3, {0}, &miss_keys_ptr));
      return;
    }

    Tensor miss_count_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_INT32, TensorShape({}), &miss_count_tensor));
    int32* d_miss_count = miss_count_tensor.flat<int32>().data();

    Tensor hit_and_miss_keys_indices;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({key_count}),
                                           &hit_and_miss_keys_indices));
    int32* d_hit_and_miss_keys_indices =
        hit_and_miss_keys_indices.flat<int32>().data();

    Tensor hit_cache_indices_and_miss_keys;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           TensorShape({key_count}),
                                           &hit_cache_indices_and_miss_keys));
    T* d_hit_cache_indices_and_miss_keys =
        hit_cache_indices_and_miss_keys.flat<T>().data();

    auto* range = ::hybridbackend::ProfilerRange::forLookup("Lookup");
    functor::LookupFunctor<int64> lookup;
    lookup(d_miss_count, d_hit_and_miss_keys_indices,
           d_hit_cache_indices_and_miss_keys, keys_cache_slab_count,
           d_keys_cache, key_count, d_keys, ctx->eigen_device<GPUDevice>());

    int32 miss_count;
    se::DeviceMemoryBase miss_count_ptr(d_miss_count, sizeof(int32));
    stream->ThenMemcpy(&miss_count, miss_count_ptr, sizeof(int32));
    stream->BlockHostUntilDone();

    ctx->set_output(0,
                    hit_and_miss_keys_indices.Slice(0, key_count - miss_count));
    ctx->set_output(
        1, hit_cache_indices_and_miss_keys.Slice(0, key_count - miss_count));
    ctx->set_output(2, hit_and_miss_keys_indices.Slice(miss_count, key_count));
    ctx->set_output(
        3, hit_cache_indices_and_miss_keys.Slice(miss_count, key_count));
    ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        ctx->op_device_context()->stream(), [range]() { delete range; });
  }

 private:
  int cache_slab_size_;
};

#define REGISTER_LOOKUP_KERNEL(T) \
  REGISTER_KERNEL_BUILDER(        \
      Name("HbLookup").Device(DEVICE_GPU).TypeConstraint<T>("T"), LookupOp<T>)
TF_CALL_int64(REGISTER_LOOKUP_KERNEL);
#undef REGISTER_LOOKUP_KERNEL

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // HYBRIDBACKEND_TENSORFLOW

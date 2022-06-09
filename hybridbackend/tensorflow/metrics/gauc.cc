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

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {
namespace hybridbackend {
namespace {
template <typename T>
T GetNonClick(T* plabels, size_t k, int dim) {
  if (dim == 1) return 1.0 - plabels[k];
  return plabels[2 * k];
}

template <typename T>
T GetClick(T* plabels, size_t k, int dim) {
  if (dim == 1) return plabels[k];
  return plabels[2 * k + 1];
}

template <typename T>
bool ComputeGauc(T* plabels, T* ppreds, T* pfilter, size_t* pidx, size_t l,
                 size_t r, int dim, double* ret) {
  std::sort(pidx + l, pidx + r, [ppreds, dim](size_t a, size_t b) {
    return GetClick<T>(ppreds, a, dim) < GetClick<T>(ppreds, b, dim);
  });
  double fp1, tp1, fp2, tp2, auc;
  fp1 = tp1 = fp2 = tp2 = auc = 0;
  size_t i;
  for (size_t k = l; k < r; ++k) {
    i = pidx[k];
    if (pfilter != nullptr && pfilter[i] == 0) continue;
    fp2 += GetNonClick<T>(plabels, i, dim);
    tp2 += GetClick<T>(plabels, i, dim);
    auc += (fp2 - fp1) * (tp2 + tp1);
    fp1 = fp2;
    tp1 = tp2;
  }
  double threshold = static_cast<double>(r - l) - 1e-3;
  if (tp2 > threshold or fp2 > threshold) {
    *ret = -0.5;
    return true;
  }
  if (tp2 * fp2 > 0) {
    *ret = (1.0 - auc / (2.0 * tp2 * fp2));
    return true;
  }
  return false;
}
}  // anonymous namespace

REGISTER_OP("HbGaucCalc")
    .Output("aucs: T")
    .Output("counts: int32")
    .Input("labels: T")
    .Input("predictions: T")
    .Input("indicators: Tindicators")
    .Attr("T: {float, double}")
    .Attr("Tindicators: {int32, int64}")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(siran.ysr) Specify more accurate shape function and add operator docs.

template <typename T, typename Tindicators>
class GaucCalcOp : public OpKernel {
 public:
  explicit GaucCalcOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& labels_t = ctx->input(0);
    const Tensor& predictions_t = ctx->input(1);
    const Tensor& indicators_t = ctx->input(2);

    size_t ldim = labels_t.shape().dims();
    size_t n = labels_t.shape().dim_size(0);
    std::vector<size_t> index(n);
    for (size_t i = 0; i < n; ++i) index[i] = i;

    T* labels = const_cast<T*>(&labels_t.flat<T>()(0));
    T* predictions = const_cast<T*>(&predictions_t.flat<T>()(0));
    auto indicators = indicators_t.flat<Tindicators>();
    std::vector<double> auc_values;
    std::vector<size_t> count_values;
    bool first = true;
    for (size_t begin = 0, end = 0; end < n; ++end) {
      if (indicators(end) == indicators(begin)) continue;

      if (first) {
        first = false;
      } else {
        double auc = 0;
        if (ComputeGauc<T>(labels, predictions, nullptr, index.data(), begin,
                           end, ldim, &auc)) {
          if (auc >= 0) {
            auc_values.emplace_back(auc);
            count_values.emplace_back(end - begin);
          }
        }
      }
      begin = end;
    }

    Tensor* aucs_t;
    Tensor* counts_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {static_cast<int64>(auc_values.size())},
                                  &aucs_t));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, {static_cast<int64>(count_values.size())},
                                  &counts_t));
    std::copy(auc_values.begin(), auc_values.end(), &aucs_t->vec<T>()(0));
    std::copy(count_values.begin(), count_values.end(),
              &counts_t->vec<int32>()(0));
  }
};

#define REGISTER_GAUC_CALC(type, indicator_type)                              \
  REGISTER_KERNEL_BUILDER(Name("HbGaucCalc")                                  \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<indicator_type>("Tindicators"), \
                          GaucCalcOp<type, indicator_type>)

REGISTER_GAUC_CALC(float, int32);
REGISTER_GAUC_CALC(float, int64);
REGISTER_GAUC_CALC(double, int32);
REGISTER_GAUC_CALC(double, int64);

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_GRENDEZ

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("BroadcastWithGrendez")
    .Output("output: T")
    .Input("input: T")
    .Input("step: int64")
    .Attr("root_rank: int >= 0 = 0")
    .Attr("shared_name: string")
    .Attr("devices: list(string) >= 2")
    .Attr("rank: int >= 0 = 0")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .SetIsStateful()
    .Doc(R"doc(
Broadcasts all inputs to devices using global rendezvous.

Ops with same `shared_name` should run in all devices at same time.

output: Output tensor across all devices.
input: Input tensor in this device to broadcast.
step: Step of collective communication, must be scalar.
root_rank: Root device rank of this op in this collective communication.
shared_name: Name of all ops in this collective communication.
devices: List of device names in this collective communication.
rank: Device rank of this op in this collective communication.
)doc");

template <typename Device, typename T>
class BroadcastWithGrendezOp : public AsyncOpKernel {
 public:
  BroadcastWithGrendezOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("devices", &devices_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("root_rank", &root_rank_));
    OP_REQUIRES(
        ctx, root_rank_ >= 0,
        errors::InvalidArgument("root_rank should not be smaller than 0"));
    OP_REQUIRES(ctx, static_cast<size_t>(root_rank_) < devices_.size(),
                errors::InvalidArgument(
                    "root_rank should be smaller than number of devices"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rank", &rank_));
    OP_REQUIRES(ctx, rank_ >= 0,
                errors::InvalidArgument("rank should not be smaller than 0"));
    OP_REQUIRES(ctx, static_cast<size_t>(rank_) < devices_.size(),
                errors::InvalidArgument(
                    "rank should be smaller than number of devices"));
  }

  virtual void ComputeAsync(OpKernelContext* ctx,
                            AsyncOpKernel::DoneCallback done) {
    Status status;
    const string actual_device = ctx->device()->attributes().name();
    const string supposed_device = devices_[rank_];
    OP_REQUIRES_ASYNC(
        ctx, actual_device == supposed_device,
        errors::InvalidArgument("devices[", rank_, "] ", supposed_device,
                                " is different to op's device ", actual_device),
        done);

    Rendezvous* rendezvous = ctx->global_rendezvous();
    OP_REQUIRES_ASYNC(ctx, rendezvous != nullptr,
                      errors::Internal("Collective needs a rendezvous"), done);

    const Tensor* input;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input));

    const Tensor* step;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("step", &step), done);
    OP_REQUIRES_ASYNC(ctx, IsLegacyScalar(step->shape()),
                      errors::InvalidArgument("step is not a scalar: ",
                                              step->shape().DebugString()),
                      done);
    const int64 step_value = step->scalar<int64>()();
    size_t num_devices = devices_.size();
    const string frame_iter = strings::StrCat(ctx->frame_iter().frame_id, ":",
                                              ctx->frame_iter().iter_id);

    if (rank_ == root_rank_) {
      const string send_key = strings::StrCat(
          devices_[rank_], ";0;", devices_[(rank_ + 1) % num_devices], ";",
          shared_name_, ":[", step_value, "]:", 0, ";", frame_iter);
      Rendezvous::ParsedKey parsed;
      OP_REQUIRES_OK_ASYNC(ctx, Rendezvous::ParseKey(send_key, &parsed), done);

      Rendezvous::Args args;
      args.device_context = ctx->op_device_context();
      args.alloc_attrs = ctx->output_alloc_attr(0);
      VLOG(1) << shared_name_ << " [" << name() << "] [Send] " << send_key;
      OP_REQUIRES_OK_ASYNC(
          ctx, rendezvous->Send(parsed, args, *input, ctx->is_input_dead()),
          done);

      if (IsRefType(ctx->input_dtype(0))) {
        ctx->forward_ref_input_to_ref_output(0, 0);
      } else {
        ctx->set_output(0, *input);
      }
      done();
      return;
    }

    const string recv_key =
        strings::StrCat(devices_[(rank_ - 1 + num_devices) % num_devices],
                        ";0;", devices_[rank_], ";", shared_name_, ":[",
                        step_value, "]:", 0, ";", frame_iter);
    Rendezvous::ParsedKey recv_parsed;
    OP_REQUIRES_OK_ASYNC(ctx, Rendezvous::ParseKey(recv_key, &recv_parsed),
                         done);

    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->output_alloc_attr(0);

    const string send_key = strings::StrCat(
        devices_[rank_], ";0;", devices_[(rank_ + 1) % num_devices], ";",
        shared_name_, ":[", step_value, "]:", 0, ";", frame_iter);
    Rendezvous::ParsedKey send_parsed;
    OP_REQUIRES_OK_ASYNC(ctx, Rendezvous::ParseKey(send_key, &send_parsed),
                         done);
    const bool has_next =
        (rank_ + 1) % static_cast<int64>(num_devices) != root_rank_;
    VLOG(1) << shared_name_ << " [" << name() << "] [Recv] " << recv_key;
    rendezvous->RecvAsync(
        recv_parsed, args,
        [this, ctx, rendezvous, done, send_parsed, send_key, recv_key,
         has_next](const Status& s, const Rendezvous::Args& send_args,
                   const Rendezvous::Args& recv_args, const Tensor& v,
                   const bool is_dead) {
          VLOG(1) << shared_name_ << " [" << name() << "] [Recv] [Done] "
                  << recv_key;
          ctx->SetStatus(s);
          if (!s.ok()) {
            done();
            return;
          }

          ctx->set_output(0, v);
          if (has_next) {
            VLOG(1) << shared_name_ << " [" << name() << "] [Send] "
                    << send_key;
            OP_REQUIRES_OK_ASYNC(
                ctx, rendezvous->Send(send_parsed, recv_args, v, is_dead),
                done);
          }

          done();
        });
  }

 protected:
  string shared_name_;
  std::vector<string> devices_;
  int64 root_rank_;
  int64 rank_;
};

#define REGISTER_KERNEL(TYPE)                             \
  REGISTER_KERNEL_BUILDER(Name("BroadcastWithGrendez")    \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          BroadcastWithGrendezOp<CPUDevice, TYPE>);
TF_CALL_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
#define REGISTER_KERNEL(TYPE)                            \
  REGISTER_KERNEL_BUILDER(Name("BroadcastWithGrendez")   \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<TYPE>("T") \
                              .HostMemory("step"),       \
                          BroadcastWithGrendezOp<GPUDevice, TYPE>);
TF_CALL_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

#endif  // HYBRIDBACKEND_GRENDEZ

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

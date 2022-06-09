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

#include <vector>

#include <absl/strings/str_cat.h>

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#if GOOGLE_CUDA
#include <tensorflow/stream_executor/cuda/cuda_activation.h>

#include "hybridbackend/common/profiler.h"
#include "hybridbackend/tensorflow/distribute/nccl/comm.h"
#endif

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL
#if GOOGLE_CUDA

NcclComm::NcclComm() {}

NcclComm::~NcclComm() {
  if (created_) {
    Destroy();
  }
}

Status NcclComm::Initialize(int size, int rank, const string& shared_name,
                            OpKernelContext* ctx) {
  if (!TF_PREDICT_TRUE(0 <= rank && rank < size)) {
    return errors::InvalidArgument(
        absl::StrCat("NCCL rank ", rank, " or size ", size, " is invalid."));
  }

  size_ = size;
  rank_ = rank;

  string thread_pool_name("nccl_comm_thread_");
  for (size_t i = 0; i < shared_name.size(); ++i) {
    const char ch = shared_name[i];
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
      thread_pool_name += ch;
    } else {
      thread_pool_name += '_';
    }
  }
  thread_pool_.reset(new thread::ThreadPool(
      ctx->env(), ThreadOptions(), thread_pool_name, 1 /* num_threads */,
      false /* low_latency_hint */));

  ctx_stream_ = new se::Stream(ctx->op_device_context()->stream()->parent());
  // NOTE(zycao): CU_STREAM_NON_BLOCKING was not supported yet.
  // Using this argument for non-blocking by default stream.
  ctx_stream_->Init();

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1010L
  stream_ = reinterpret_cast<cudaStream_t*>(
      ctx_stream_->implementation()->CudaStreamMemberHack());
#else
  stream_ = reinterpret_cast<cudaStream_t*>(
      ctx_stream_->implementation()->GpuStreamMemberHack());
#endif

  debug_string_ = absl::StrCat("NcclComm(name=", shared_name, ", size=", size,
                               ", rank=", rank, ")");

  return Status::OK();
}

Status NcclComm::Create(const string& id) {
  if (!TF_PREDICT_TRUE(id.size() == NCCL_UNIQUE_ID_BYTES)) {
    return errors::InvalidArgument(
        absl::StrCat("NCCL ID ", id.c_str(), " is invalid."));
  }

  ncclUniqueId nccl_id;
  memcpy(nccl_id.internal, &id[0], NCCL_UNIQUE_ID_BYTES);
  TF_RETURN_IF_ERROR(
      NcclErrorToStatus(ncclCommInitRank(&comm_, size_, nccl_id, rank_)));
  created_ = true;

  return Status::OK();
}

Status NcclComm::Destroy() {
  TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclCommDestroy(comm_)));
  created_ = false;
  return Status::OK();
}

Status NcclComm::Abort() {
  TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclCommAbort(comm_)));
  created_ = false;
  return Status::OK();
}

Status NcclComm::UserRank(int* rank) {
  return NcclErrorToStatus(ncclCommUserRank(comm_, rank));
}

void NcclComm::RunAsync(const string& message, OpKernelContext* ctx,
                        AsyncOpKernel::DoneCallback done,
                        std::function<void()> func) {
  int device_id;
  cudaGetDevice(&device_id);
  se::Event* inputs_ready =
      new se::Event(ctx->op_device_context()->stream()->parent());
  inputs_ready->Init();
  ctx->op_device_context()->stream()->ThenRecordEvent(inputs_ready);
  thread_pool_->Schedule(
      [device_id, inputs_ready, func, this, ctx, done, message]() {
        cudaSetDevice(device_id);
        se::cuda::ScopedActivateExecutorContext context(ctx_stream_->parent());
        this->ThenWaitFor(inputs_ready);
        auto* range = ::hybridbackend::ProfilerRange::forSynch(message);
        func();
        auto outputs_ready = this->ThenRecordEvent();
        ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
            ctx->op_device_context()->stream(),
            [outputs_ready, ctx, done, range]() {
              ctx->op_device_context()->stream()->ThenWaitFor(outputs_ready);
              delete outputs_ready;
              delete range;
              done();
            });
      });
}

se::Event* NcclComm::ThenRecordEvent() {
  se::Event* ev = new se::Event(ctx_stream_->parent());
  ev->Init();
  ctx_stream_->ThenRecordEvent(ev);
  return ev;
}

void NcclComm::ThenWaitFor(se::Event* ev) {
  ctx_stream_->ThenWaitFor(ev);
  delete ev;
}

void NcclComm::BlockHostUntilDone() { ctx_stream_->BlockHostUntilDone(); }

Status NcclComm::Reduce(const Tensor& input, const ncclRedOp_t reduce_op,
                        const int root_rank, Tensor* output) {
  const void* sendbuf = input.tensor_data().data();
  void* recvbuf = const_cast<char*>(output->tensor_data().data());
  const size_t count = input.NumElements();
  ncclDataType_t nccl_dtype(ncclFloat);
  TF_RETURN_IF_ERROR(EnumToNcclEnum(input.dtype(), &nccl_dtype));

  return NcclErrorToStatus(ncclReduce(sendbuf, recvbuf, count, nccl_dtype,
                                      reduce_op, root_rank, comm_, *stream_));
}

Status NcclComm::ReduceScatter(const Tensor& input, const ncclRedOp_t reduce_op,
                               Tensor* output) {
  const void* sendbuf = input.tensor_data().data();
  void* recvbuf = const_cast<char*>(output->tensor_data().data());
  const size_t count = output->NumElements();
  ncclDataType_t nccl_dtype(ncclFloat);
  TF_RETURN_IF_ERROR(EnumToNcclEnum(input.dtype(), &nccl_dtype));

  return NcclErrorToStatus(ncclReduceScatter(
      sendbuf, recvbuf, count, nccl_dtype, reduce_op, comm_, *stream_));
}

Status NcclComm::Allreduce(const Tensor& input, const ncclRedOp_t reduce_op,
                           Tensor* output) {
  const void* sendbuf = input.tensor_data().data();
  void* recvbuf = const_cast<char*>(output->tensor_data().data());
  const size_t count = input.NumElements();
  ncclDataType_t nccl_dtype(ncclFloat);
  TF_RETURN_IF_ERROR(EnumToNcclEnum(input.dtype(), &nccl_dtype));

  return NcclErrorToStatus(ncclAllReduce(sendbuf, recvbuf, count, nccl_dtype,
                                         reduce_op, comm_, *stream_));
}

Status NcclComm::Broadcast(const Tensor& input, const int root_rank,
                           Tensor* output) {
  const void* sendbuf = input.tensor_data().data();
  void* recvbuf = const_cast<char*>(output->tensor_data().data());
  const size_t count = input.NumElements();
  ncclDataType_t nccl_dtype(ncclFloat);
  TF_RETURN_IF_ERROR(EnumToNcclEnum(input.dtype(), &nccl_dtype));

  return NcclErrorToStatus(ncclBroadcast(sendbuf, recvbuf, count, nccl_dtype,
                                         root_rank, comm_, *stream_));
}

Status NcclComm::Allgather(const Tensor& input, Tensor* output) {
  const void* sendbuf = input.tensor_data().data();
  void* recvbuf = const_cast<char*>(output->tensor_data().data());
  const size_t count = input.NumElements();
  ncclDataType_t nccl_dtype(ncclFloat);
  TF_RETURN_IF_ERROR(EnumToNcclEnum(input.dtype(), &nccl_dtype));

  return NcclErrorToStatus(
      ncclAllGather(sendbuf, recvbuf, count, nccl_dtype, comm_, *stream_));
}

Status NcclComm::GroupAllgather(const std::vector<Tensor>& inputs,
                                Tensor* output) {
  ncclGroupStart();
  size_t offset = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const void* sendbuf = inputs[i].tensor_data().data();
    void* recvbuf = const_cast<char*>(output->tensor_data().data()) + offset;
    const size_t count = inputs[i].NumElements();
    ncclDataType_t nccl_dtype(ncclFloat);
    TF_RETURN_IF_ERROR(EnumToNcclEnum(inputs[i].dtype(), &nccl_dtype));
    TF_RETURN_IF_ERROR(NcclErrorToStatus(
        ncclAllGather(sendbuf, recvbuf, count, nccl_dtype, comm_, *stream_)));
    offset += size_ * count * DataTypeSize(inputs[i].dtype());
  }
  ncclGroupEnd();
  return Status::OK();
}

Status NcclComm::Allgatherv(const Tensor& input, const Tensor& host_sizes,
                            Tensor* output) {
  const void* sendbuf = input.tensor_data().data();
  char* recvbuf = const_cast<char*>(output->tensor_data().data());
  const int32* sendcounts = host_sizes.flat<int32>().data();
  ncclDataType_t nccl_dtype(ncclFloat);
  TF_RETURN_IF_ERROR(EnumToNcclEnum(input.dtype(), &nccl_dtype));
  const size_t dtype_size = DataTypeSize(input.dtype());

  size_t offset = 0;
  ncclGroupStart();
  for (int i = 0; i < size_; ++i) {
    TF_RETURN_IF_ERROR(NcclErrorToStatus(
        ncclBroadcast(sendbuf, recvbuf + offset, sendcounts[i], nccl_dtype, i,
                      comm_, *stream_)));
    offset += sendcounts[i] * dtype_size;
  }
  ncclGroupEnd();
  return Status::OK();
}

Status NcclComm::Alltoall(const Tensor& input, Tensor* output) {
  const char* sendbuf = input.tensor_data().data();
  char* recvbuf = const_cast<char*>(output->tensor_data().data());
  const size_t count = input.NumElements();
  if (TF_PREDICT_FALSE(count % size_ != 0)) {
    return errors::InvalidArgument("Number of elements in input (", count,
                                   ") must can be divided into ", size_,
                                   " partitions");
  }
  const size_t partition_size = count / size_;
  ncclDataType_t nccl_dtype(ncclFloat);
  TF_RETURN_IF_ERROR(EnumToNcclEnum(input.dtype(), &nccl_dtype));
  const size_t dtype_size = DataTypeSize(input.dtype());

#if NCCL_VERSION_CODE >= 2700
  ncclGroupStart();
  for (int i = 0; i < size_; ++i) {
    const size_t partition_bytes = i * partition_size * dtype_size;
    TF_RETURN_IF_ERROR(
        NcclErrorToStatus(ncclSend(sendbuf + partition_bytes, partition_size,
                                   nccl_dtype, i, comm_, *stream_)));
    TF_RETURN_IF_ERROR(
        NcclErrorToStatus(ncclRecv(recvbuf + partition_bytes, partition_size,
                                   nccl_dtype, i, comm_, *stream_)));
  }
  ncclGroupEnd();
  return Status::OK();
#else
  return errors::Unimplemented("Alltoall not supported in NCCL < 2.7");
#endif
}

Status NcclComm::Alltoallv(const Tensor& input, const Tensor& host_sizes,
                           const int64 common_size, Tensor* output) {
  const char* sendbuf = input.tensor_data().data();
  char* recvbuf = const_cast<char*>(output->tensor_data().data());
  const int32* sendcounts = host_sizes.flat<int32>().data();
  ncclDataType_t nccl_dtype(ncclFloat);
  TF_RETURN_IF_ERROR(EnumToNcclEnum(input.dtype(), &nccl_dtype));
  const int32 dtype_size = static_cast<int32>(DataTypeSize(input.dtype()));

#if NCCL_VERSION_CODE >= 2700
  int32 sendoffset = 0;
  int32 recvoffset = 0;
  ncclGroupStart();
  for (int i = 0; i < size_; ++i) {
    const int32 sendsize = sendcounts[size_ * rank_ + i] * common_size;
    const int32 recvsize = sendcounts[size_ * i + rank_] * common_size;
    TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclSend(
        sendbuf + sendoffset, sendsize, nccl_dtype, i, comm_, *stream_)));
    TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclRecv(
        recvbuf + recvoffset, recvsize, nccl_dtype, i, comm_, *stream_)));
    sendoffset += sendsize * dtype_size;
    recvoffset += recvsize * dtype_size;
  }
  ncclGroupEnd();
  return Status::OK();
#else
  return errors::Unimplemented("Alltoallv not supported in NCCL < 2.7");
#endif
}

Status NcclComm::AlltoallvN(const std::vector<Tensor>& inputs,
                            const Tensor& host_sizes,
                            const std::vector<int64>& common_sizes,
                            std::vector<Tensor*>* outputs) {
  if (TF_PREDICT_FALSE(outputs->size() != inputs.size() ||
                       host_sizes.NumElements() !=
                           inputs.size() * size_ * size_)) {
    return errors::InvalidArgument(
        "Size of inputs and size of outputs must be same and host_sizes must "
        "have all sizes for all inputs across all devices");
  }

#if NCCL_VERSION_CODE >= 2700
  ncclGroupStart();
  for (size_t gid = 0; gid < inputs.size(); ++gid) {
    const char* sendbuf = inputs[gid].tensor_data().data();
    char* recvbuf = const_cast<char*>(outputs->at(gid)->tensor_data().data());
    const int32* sendcounts =
        host_sizes.flat<int32>().data() + gid * size_ * size_;
    ncclDataType_t nccl_dtype(ncclFloat);
    TF_RETURN_IF_ERROR(EnumToNcclEnum(inputs[gid].dtype(), &nccl_dtype));
    const int32 dtype_size =
        static_cast<int32>(DataTypeSize(inputs[gid].dtype()));

    int32 sendoffset = 0;
    int32 recvoffset = 0;
    ncclGroupStart();
    for (int i = 0; i < size_; ++i) {
      const int32 sendsize = sendcounts[size_ * rank_ + i] * common_sizes[gid];
      const int32 recvsize = sendcounts[size_ * i + rank_] * common_sizes[gid];
      TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclSend(
          sendbuf + sendoffset, sendsize, nccl_dtype, i, comm_, *stream_)));
      TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclRecv(
          recvbuf + recvoffset, recvsize, nccl_dtype, i, comm_, *stream_)));
      sendoffset += sendsize * dtype_size;
      recvoffset += recvsize * dtype_size;
    }
    ncclGroupEnd();
  }
  ncclGroupEnd();
  return Status::OK();
#else
  return errors::Unimplemented("AlltoallvN not supported in NCCL < 2.7");
#endif
}

Status NcclComm::AlltoallvN(const std::vector<Tensor*>& inputs,
                            const Tensor& host_sizes,
                            const std::vector<int64>& common_sizes,
                            std::vector<Tensor*>* outputs) {
  if (TF_PREDICT_FALSE(outputs->size() != inputs.size() ||
                       host_sizes.NumElements() !=
                           inputs.size() * size_ * size_)) {
    return errors::InvalidArgument(
        "Size of inputs and size of outputs must be same and host_sizes must "
        "have all sizes for all inputs across all devices");
  }

#if NCCL_VERSION_CODE >= 2700
  ncclGroupStart();
  for (size_t gid = 0; gid < inputs.size(); ++gid) {
    const char* sendbuf = inputs[gid]->tensor_data().data();
    char* recvbuf = const_cast<char*>(outputs->at(gid)->tensor_data().data());
    const int32* sendcounts =
        host_sizes.flat<int32>().data() + gid * size_ * size_;
    ncclDataType_t nccl_dtype(ncclFloat);
    TF_RETURN_IF_ERROR(EnumToNcclEnum(inputs[gid]->dtype(), &nccl_dtype));
    const int32 dtype_size =
        static_cast<int32>(DataTypeSize(inputs[gid]->dtype()));

    int32 sendoffset = 0;
    int32 recvoffset = 0;
    ncclGroupStart();
    for (int i = 0; i < size_; ++i) {
      const int32 sendsize = sendcounts[size_ * rank_ + i] * common_sizes[gid];
      const int32 recvsize = sendcounts[size_ * i + rank_] * common_sizes[gid];
      TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclSend(
          sendbuf + sendoffset, sendsize, nccl_dtype, i, comm_, *stream_)));
      TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclRecv(
          recvbuf + recvoffset, recvsize, nccl_dtype, i, comm_, *stream_)));
      sendoffset += sendsize * dtype_size;
      recvoffset += recvsize * dtype_size;
    }
    ncclGroupEnd();
  }
  ncclGroupEnd();
  return Status::OK();
#else
  return errors::Unimplemented("AlltoallvN not supported in NCCL < 2.7");
#endif
}

Status NcclComm::Alltoallw(const std::vector<Tensor>& inputs,
                           std::vector<Tensor*>* outputs) {
  if (TF_PREDICT_FALSE(inputs.size() != size_ || outputs->size() != size_)) {
    return errors::InvalidArgument(
        "Size of inputs and outputs must be equal to the communicator size");
  }
  std::vector<const char*> sendbufs;
  std::vector<char*> recvbufs;
  std::vector<size_t> sendcounts;
  std::vector<size_t> recvcounts;
  std::vector<ncclDataType_t> senddtypes;
  std::vector<ncclDataType_t> recvdtypes;
  for (int i = 0; i < size_; ++i) {
    sendbufs.push_back(inputs[i].tensor_data().data());
    recvbufs.push_back(const_cast<char*>((*outputs)[i]->tensor_data().data()));
    sendcounts.push_back(inputs[i].NumElements());
    recvcounts.push_back((*outputs)[i]->NumElements());
    ncclDataType_t senddtype;
    TF_RETURN_IF_ERROR(EnumToNcclEnum(inputs[i].dtype(), &senddtype));
    senddtypes.push_back(senddtype);
    ncclDataType_t recvdtype;
    TF_RETURN_IF_ERROR(EnumToNcclEnum((*outputs)[i]->dtype(), &recvdtype));
    recvdtypes.push_back(recvdtype);
  }

#if NCCL_VERSION_CODE >= 2700
  ncclGroupStart();
  for (int i = 0; i < size_; ++i) {
    if (rank_ == i) {
      if (TF_PREDICT_TRUE(sendbufs[i] == recvbufs[i])) {
        continue;
      }
    }
    TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclSend(
        sendbufs[i], sendcounts[i], senddtypes[i], i, comm_, *stream_)));
    TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclRecv(
        recvbufs[i], recvcounts[i], recvdtypes[i], i, comm_, *stream_)));
  }
  ncclGroupEnd();
  return Status::OK();
#else
  return errors::Unimplemented("Alltoallw not supported in NCCL < 2.7");
#endif
}

Status NcclComm::Alltoallw(const std::vector<Tensor*>& inputs,
                           std::vector<Tensor*>* outputs) {
  if (TF_PREDICT_FALSE(inputs.size() != size_ || outputs->size() != size_)) {
    return errors::InvalidArgument(
        "Size of inputs and outputs must be equal to the communicator size");
  }
  std::vector<const char*> sendbufs;
  std::vector<char*> recvbufs;
  std::vector<size_t> sendcounts;
  std::vector<size_t> recvcounts;
  std::vector<ncclDataType_t> senddtypes;
  std::vector<ncclDataType_t> recvdtypes;
  for (int i = 0; i < size_; ++i) {
    sendbufs.push_back(inputs[i]->tensor_data().data());
    recvbufs.push_back(const_cast<char*>((*outputs)[i]->tensor_data().data()));
    sendcounts.push_back(inputs[i]->NumElements());
    recvcounts.push_back((*outputs)[i]->NumElements());
    ncclDataType_t senddtype;
    TF_RETURN_IF_ERROR(EnumToNcclEnum(inputs[i]->dtype(), &senddtype));
    senddtypes.push_back(senddtype);
    ncclDataType_t recvdtype;
    TF_RETURN_IF_ERROR(EnumToNcclEnum((*outputs)[i]->dtype(), &recvdtype));
    recvdtypes.push_back(recvdtype);
  }

#if NCCL_VERSION_CODE >= 2700
  ncclGroupStart();
  for (int i = 0; i < size_; ++i) {
    if (rank_ == i) {
      if (TF_PREDICT_TRUE(sendbufs[i] == recvbufs[i])) {
        continue;
      }
    }
    TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclSend(
        sendbufs[i], sendcounts[i], senddtypes[i], i, comm_, *stream_)));
    TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclRecv(
        recvbufs[i], recvcounts[i], recvdtypes[i], i, comm_, *stream_)));
  }
  ncclGroupEnd();
  return Status::OK();
#else
  return errors::Unimplemented("Alltoallw not supported in NCCL < 2.7");
#endif
}

Status NcclComm::AlltoallwN(const std::vector<Tensor>& inputs,
                            const int64 num_columns,
                            std::vector<Tensor*>* outputs) {
  if (TF_PREDICT_FALSE(inputs.size() != num_columns * size_ ||
                       outputs->size() != num_columns * size_)) {
    return errors::InvalidArgument(
        "Size of send_tensors and recv_tensors must be num_columns times of "
        "size of the communicator");
  }

  std::vector<const char*> sendbufs;
  std::vector<char*> recvbufs;
  std::vector<size_t> sendcounts;
  std::vector<size_t> recvcounts;
  std::vector<ncclDataType_t> senddtypes;
  std::vector<ncclDataType_t> recvdtypes;
  for (int i = 0; i < inputs.size(); ++i) {
    sendbufs.push_back(inputs[i].tensor_data().data());
    recvbufs.push_back(const_cast<char*>((*outputs)[i]->tensor_data().data()));
    sendcounts.push_back(inputs[i].NumElements());
    recvcounts.push_back((*outputs)[i]->NumElements());
    ncclDataType_t senddtype;
    TF_RETURN_IF_ERROR(EnumToNcclEnum(inputs[i].dtype(), &senddtype));
    senddtypes.push_back(senddtype);
    ncclDataType_t recvdtype;
    TF_RETURN_IF_ERROR(EnumToNcclEnum((*outputs)[i]->dtype(), &recvdtype));
    recvdtypes.push_back(recvdtype);
  }

#if NCCL_VERSION_CODE >= 2700
  ncclGroupStart();
  for (int g = 0; g < num_columns; g++) {
    ncclGroupStart();
    for (int i = 0; i < size_; i++) {
      int idx = g * size_ + i;
      if (rank_ == i) {
        if (TF_PREDICT_TRUE(sendbufs[idx] == recvbufs[idx])) {
          continue;
        }
      }
      TF_RETURN_IF_ERROR(
          NcclErrorToStatus(ncclSend(sendbufs[idx], sendcounts[idx],
                                     senddtypes[idx], i, comm_, *stream_)));
      TF_RETURN_IF_ERROR(
          NcclErrorToStatus(ncclRecv(recvbufs[idx], recvcounts[idx],
                                     recvdtypes[idx], i, comm_, *stream_)));
    }
    ncclGroupEnd();
  }
  ncclGroupEnd();
  return Status::OK();
#else
  return errors::Unimplemented("AlltoallwN not supported in NCCL < 2.7");
#endif
}

Status NcclComm::AlltoallwN(const std::vector<Tensor*>& inputs,
                            const int64 num_columns,
                            std::vector<Tensor*>* outputs) {
  if (TF_PREDICT_FALSE(inputs.size() != num_columns * size_ ||
                       outputs->size() != num_columns * size_)) {
    return errors::InvalidArgument(
        "Size of send_tensors and recv_tensors must be num_columns times of "
        "size of the communicator");
  }

  std::vector<const char*> sendbufs;
  std::vector<char*> recvbufs;
  std::vector<size_t> sendcounts;
  std::vector<size_t> recvcounts;
  std::vector<ncclDataType_t> senddtypes;
  std::vector<ncclDataType_t> recvdtypes;
  for (int i = 0; i < inputs.size(); ++i) {
    sendbufs.push_back(inputs[i]->tensor_data().data());
    recvbufs.push_back(const_cast<char*>((*outputs)[i]->tensor_data().data()));
    sendcounts.push_back(inputs[i]->NumElements());
    recvcounts.push_back((*outputs)[i]->NumElements());
    ncclDataType_t senddtype;
    TF_RETURN_IF_ERROR(EnumToNcclEnum(inputs[i]->dtype(), &senddtype));
    senddtypes.push_back(senddtype);
    ncclDataType_t recvdtype;
    TF_RETURN_IF_ERROR(EnumToNcclEnum((*outputs)[i]->dtype(), &recvdtype));
    recvdtypes.push_back(recvdtype);
  }

#if NCCL_VERSION_CODE >= 2700
  ncclGroupStart();
  for (int g = 0; g < num_columns; g++) {
    ncclGroupStart();
    for (int i = 0; i < size_; i++) {
      int idx = g * size_ + i;
      if (rank_ == i) {
        if (TF_PREDICT_TRUE(sendbufs[idx] == recvbufs[idx])) {
          continue;
        }
      }
      TF_RETURN_IF_ERROR(
          NcclErrorToStatus(ncclSend(sendbufs[idx], sendcounts[idx],
                                     senddtypes[idx], i, comm_, *stream_)));
      TF_RETURN_IF_ERROR(
          NcclErrorToStatus(ncclRecv(recvbufs[idx], recvcounts[idx],
                                     recvdtypes[idx], i, comm_, *stream_)));
    }
    ncclGroupEnd();
  }
  ncclGroupEnd();
  return Status::OK();
#else
  return errors::Unimplemented("AlltoallwN not supported in NCCL < 2.7");
#endif
}

NcclCommAsyncOp::NcclCommAsyncOp(OpKernelConstruction* ctx)
    : AsyncOpKernel(ctx) {}

void NcclCommAsyncOp::ComputeAsync(OpKernelContext* ctx,
                                   AsyncOpKernel::DoneCallback done) {
  NcclComm* comm = nullptr;
  OP_REQUIRES_OK_ASYNC(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &comm),
                       done);
  ComputeAsyncWithComm(comm, ctx, done);
};

se::Event* NcclCommAsyncOp::ThenRecordEvent(OpKernelContext* ctx) {
  se::Event* ev = new se::Event(ctx->op_device_context()->stream()->parent());
  ev->Init();
  ctx->op_device_context()->stream()->ThenRecordEvent(ev);
  return ev;
}

void NcclCommAsyncOp::ThenWaitFor(OpKernelContext* ctx, se::Event* ev) {
  ctx->op_device_context()->stream()->ThenWaitFor(ev);
  delete ev;
}

void NcclCommAsyncOp::ThenExecute(OpKernelContext* ctx,
                                  std::function<void()> func) {
  ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
      ctx->op_device_context()->stream(), func);
}

void NcclCommAsyncOp::ThenCopyToDevice(OpKernelContext* ctx, Tensor* dst,
                                       const Tensor& src) {
  se::DeviceMemoryBase dst_ptr(const_cast<char*>(dst->tensor_data().data()),
                               dst->TotalBytes());
  ctx->op_device_context()->stream()->ThenMemcpy(
      &dst_ptr, src.tensor_data().data(), src.TotalBytes());
}

void NcclCommAsyncOp::ThenCopyToHost(OpKernelContext* ctx, Tensor* dst,
                                     const Tensor& src) {
  se::DeviceMemoryBase src_ptr(const_cast<char*>(src.tensor_data().data()),
                               src.TotalBytes());
  ctx->op_device_context()->stream()->ThenMemcpy(
      const_cast<char*>(dst->tensor_data().data()), src_ptr, dst->TotalBytes());
}

void NcclCommAsyncOp::BlockHostUntilDone(OpKernelContext* ctx) {
  ctx->op_device_context()->stream()->BlockHostUntilDone();
}

#endif
#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif  // HYBRIDBACKEND_TENSORFLOW

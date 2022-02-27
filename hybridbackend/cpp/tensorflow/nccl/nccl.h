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

#ifndef HYBRIDBACKEND_CPP_TENSORFLOW_NCCL_NCCL_H_
#define HYBRIDBACKEND_CPP_TENSORFLOW_NCCL_NCCL_H_

#if HYBRIDBACKEND_TENSORFLOW

#if GOOGLE_CUDA
#if HYBRIDBACKEND_NCCL
#include <nccl.h>
#endif
#endif

#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/core/errors.h>

namespace tensorflow {
namespace hybridbackend {

#if HYBRIDBACKEND_NCCL
#if GOOGLE_CUDA

template <typename TYPE>
struct DataTypeToNcclEnum {
  static constexpr ncclDataType_t value = ncclFloat;
};

template <ncclDataType_t VALUE>
struct NcclEnumToDataType {
  typedef float Type;
};

#define MATCH_TYPE_AND_NCCL_ENUM(TYPE, ENUM)      \
  template <>                                     \
  struct DataTypeToNcclEnum<TYPE> {               \
    static constexpr ncclDataType_t value = ENUM; \
  };                                              \
  template <>                                     \
  struct NcclEnumToDataType<ENUM> {               \
    typedef TYPE Type;                            \
  }

MATCH_TYPE_AND_NCCL_ENUM(int8, ncclInt8);
MATCH_TYPE_AND_NCCL_ENUM(uint8, ncclUint8);
MATCH_TYPE_AND_NCCL_ENUM(int32, ncclInt32);
MATCH_TYPE_AND_NCCL_ENUM(uint32, ncclUint32);
MATCH_TYPE_AND_NCCL_ENUM(int64, ncclInt64);
MATCH_TYPE_AND_NCCL_ENUM(uint64, ncclUint64);
MATCH_TYPE_AND_NCCL_ENUM(Eigen::half, ncclFloat16);
MATCH_TYPE_AND_NCCL_ENUM(float, ncclFloat32);
MATCH_TYPE_AND_NCCL_ENUM(double, ncclFloat64);

#define TF_CALL_NCCL_TYPES(m)                                             \
  TF_CALL_int8(m) TF_CALL_uint8(m) TF_CALL_int32(m) TF_CALL_uint32(m)     \
      TF_CALL_int64(m) TF_CALL_uint64(m) TF_CALL_half(m) TF_CALL_float(m) \
          TF_CALL_double(m)

inline Status NcclErrorToStatus(ncclResult_t rc) {
  if (!TF_PREDICT_TRUE(ncclSuccess == rc)) {
    return errors::Internal(ncclGetErrorString(rc));
  }
  return Status::OK();
}

inline Status EnumToNcclEnum(const DataType& dtype,
                             ncclDataType_t* nccl_dtype) {
  switch (dtype) {
    case DT_INT8:
      *nccl_dtype = ncclInt8;
      return Status::OK();
    case DT_UINT8:
      *nccl_dtype = ncclUint8;
      return Status::OK();
    case DT_INT32:
      *nccl_dtype = ncclInt32;
      return Status::OK();
    case DT_UINT32:
      *nccl_dtype = ncclUint32;
      return Status::OK();
    case DT_INT64:
      *nccl_dtype = ncclInt64;
      return Status::OK();
    case DT_UINT64:
      *nccl_dtype = ncclUint64;
      return Status::OK();
    case DT_HALF:
      *nccl_dtype = ncclFloat16;
      return Status::OK();
    case DT_FLOAT:
      *nccl_dtype = ncclFloat32;
      return Status::OK();
    case DT_DOUBLE:
      *nccl_dtype = ncclFloat64;
      return Status::OK();
    default:
      return errors::Unimplemented("Data type ", DataTypeString(dtype),
                                   " has no NCCL counterpart");
  }
}

inline Status ReduceOpToNcclReduceOp(const int reduce_op,
                                     ncclRedOp_t* nccl_reduce_op) {
  switch (reduce_op) {
    case 0:
      *nccl_reduce_op = ncclSum;
      return Status::OK();
    case 1:
      *nccl_reduce_op = ncclProd;
      return Status::OK();
    case 2:
      *nccl_reduce_op = ncclMax;
      return Status::OK();
    case 3:
      *nccl_reduce_op = ncclMin;
      return Status::OK();
#if NCCL_VERSION_CODE >= 21000
    case 4:
      *nccl_reduce_op = ncclAvg;
      return Status::OK();
#endif
    default:
      return errors::Unimplemented("Reduce op ", reduce_op,
                                   " has no NCCL counterpart");
  }
}

#endif
#endif

}  // namespace hybridbackend
}  // namespace tensorflow

#endif

#endif  // HYBRIDBACKEND_CPP_TENSORFLOW_NCCL_NCCL_H_

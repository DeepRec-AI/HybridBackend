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

// Do not report compilation warnings of tensorflow dataset implementation.
#pragma GCC system_header

#ifndef HYBRIDBACKEND_CPP_TENSORFLOW_IO_DATASET_H_
#define HYBRIDBACKEND_CPP_TENSORFLOW_IO_DATASET_H_

#if HYBRIDBACKEND_TENSORFLOW

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/public/version.h"

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) < 1015L
#define PARSE_SCALAR ParseScalarArgument
#define PARSE_VECTOR ParseVectorArgument
#else
#include "tensorflow/core/framework/dataset.h"
#define PARSE_SCALAR ::tensorflow::data::ParseScalarArgument
#define PARSE_VECTOR ::tensorflow::data::ParseVectorArgument
#endif

#endif  // HYBRIDBACKEND_TENSORFLOW

#endif  // HYBRIDBACKEND_CPP_TENSORFLOW_IO_DATASET_H_

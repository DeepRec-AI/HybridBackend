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

#ifndef HYBRIDBACKEND_COMMON_LOGGING_H_
#define HYBRIDBACKEND_COMMON_LOGGING_H_

#include <sstream>

#include "hybridbackend/common/macros.h"

#define HB_LOG_IS_ON(lvl) ((lvl) <= ::hybridbackend::MinLogLevel())

#define HB_LOG(lvl)                        \
  if (HB_PREDICT_FALSE(HB_LOG_IS_ON(lvl))) \
  ::hybridbackend::LogMessage(__FILE__, __LINE__)

namespace hybridbackend {

int& MinLogLevel();

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line);
  ~LogMessage();

 private:
  const char* fname_;
  int line_;
};

}  // namespace hybridbackend

#endif  // HYBRIDBACKEND_COMMON_LOGGING_H_

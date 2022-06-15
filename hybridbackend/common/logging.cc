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

#include "hybridbackend/common/logging.h"

#include <cstdio>
#include <ctime>

#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "hybridbackend/common/env.h"

namespace hybridbackend {

int& MinLogLevel() {
  static int* min_log_level = new int(EnvVarGetInt("HB_MIN_LOG_LEVEL", 0));
  return *min_log_level;
}

LogMessage::LogMessage(const char* fname, int line)
    : fname_(fname), line_(line) {}

LogMessage::~LogMessage() {
  static size_t pid = static_cast<size_t>(getpid());
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  struct tm rslt;
  struct tm* p = gmtime_r(&tv.tv_sec, &rslt);
  fprintf(stderr, "[%04d-%02d-%02d %02d:%02d:%02d.%ld] [%ld#%ld] [%s:%d] %s\n",
          1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min,
          p->tm_sec, tv.tv_usec, pid, syscall(SYS_gettid), fname_, line_,
          str().c_str());
}

}  // namespace hybridbackend

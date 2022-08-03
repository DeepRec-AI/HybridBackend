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

#include "hybridbackend/common/env.h"

#include <sstream>
#include <string>

#include <unistd.h>
#if HYBRIDBACKEND_CUDA
#include <cuda_runtime.h>
#endif
#include <curl/curl.h>

#include "hybridbackend/common/logging.h"

namespace hybridbackend {

std::string EnvVarGet(const std::string& env_var,
                      const std::string& default_val) {
  const char* env_var_val = getenv(env_var.c_str());
  if (env_var_val == nullptr) {
    return default_val;
  }

  std::string result(env_var_val);
  return result;
}

int EnvVarGetInt(const std::string& env_var, const int default_val) {
  const char* env_var_val = getenv(env_var.c_str());
  if (env_var_val == nullptr) {
    return default_val;
  }

  std::string env_var_val_str(env_var_val);
  std::istringstream ss(env_var_val_str);
  int result;
  if (!(ss >> result)) {
    result = default_val;
  }

  return result;
}

bool EnvVarGetBool(const std::string& env_var, const bool default_val) {
  const int r = EnvVarGetInt(env_var, default_val ? 1 : 0);
  return r != 0;
}

namespace {
size_t HttpWriteToString(void* ptr, size_t size, size_t nmemb, std::string* s) {
  s->append(static_cast<char*>(ptr), size * nmemb);
  return size * nmemb;
}
}  // namespace

std::string EnvHttpGet(const std::string& url, const std::string& default_val,
                       const long timeout) {
  CURL* curl = curl_easy_init();
  if (HB_PREDICT_FALSE(curl == nullptr)) {
    HB_LOG(0) << "[ERROR] CURL initialize failed: " << url;
    return default_val;
  }

  std::string result;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, timeout);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, HttpWriteToString);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

  CURLcode rc = curl_easy_perform(curl);
  if (HB_PREDICT_FALSE(CURLE_OK != rc)) {
    curl_easy_cleanup(curl);
    return default_val;
  }
  curl_easy_cleanup(curl);
  return result;
}

int EnvHttpGetInt(const std::string& url, const int default_val,
                  const long timeout) {
  std::string val_str = EnvHttpGet(url, "", timeout);
  if (HB_PREDICT_FALSE(val_str.empty())) {
    return default_val;
  }

  std::istringstream ss(val_str);
  int result;
  if (!(ss >> result)) {
    result = default_val;
  }

  return result;
}

std::string EnvGetInstanceType(const long timeout) {
  static std::string kUrl =
      "http://100.100.100.200/latest/meta-data/instance/instance-type";

  return EnvHttpGet(kUrl, "", timeout);
}

int EnvGetGpuCount() {
#if HYBRIDBACKEND_CUDA
  int gpu_count = 0;
  cudaError_t rc;
  rc = cudaGetDeviceCount(&gpu_count);
  if (HB_PREDICT_FALSE(cudaSuccess != rc)) {
    HB_LOG(1) << "[ERROR] Failed to query GPU count: "
              << cudaGetErrorString(rc);
    return 0;
  }

  int dev;
  rc = cudaGetDevice(&dev);
  if (HB_PREDICT_FALSE(cudaSuccess != rc)) {
    HB_LOG(0) << "[ERROR] Failed to query GPU: " << cudaGetErrorString(rc);
    return 0;
  }

  cudaDeviceProp gpu_prop;
  rc = cudaGetDeviceProperties(&gpu_prop, dev);
  if (HB_PREDICT_FALSE(cudaSuccess != rc)) {
    HB_LOG(0) << "[ERROR] Failed to query GPU properties: "
              << cudaGetErrorString(rc);
    return 0;
  }

  std::string arch = std::to_string(gpu_prop.major * 10 + gpu_prop.minor);
  std::stringstream gencode_ss("" HYBRIDBACKEND_CUDA_GENCODE "");
  std::string gencode;
  bool arch_mismatch = true;
  while (std::getline(gencode_ss, gencode, ',')) {
    if (gencode == arch) {
      arch_mismatch = false;
      break;
    }
  }
  if (HB_PREDICT_FALSE(arch_mismatch)) {
    HB_LOG(0) << "[ERROR] Failed to match GPU architecture: sm_" << arch
              << " not supported (" HYBRIDBACKEND_CUDA_GENCODE ")";
    return 0;
  }

  return gpu_count;
#else
  return 0;
#endif
}

}  // namespace hybridbackend

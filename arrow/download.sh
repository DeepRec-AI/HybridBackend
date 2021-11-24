#!/bin/bash
# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

set -eo pipefail

## Check URLs.
#ARROW_3PTY=arrow/src/cpp/cmake_modules/ThirdpartyToolchain.cmake
#URLS=$(cat $ARROW_3PTY | grep 'DEFINED ENV{ARROW.*URL' | sed 's/.*ENV{\(.*\)}.*/\1/g')

if [[ ! -d $CACHE_DIR ]]; then
  CACHE_DIR=arrow/cache
fi
mkdir -p $CACHE_DIR

if [[ ! -z "$1" ]]; then
  export ARROW_CACHE_URL_PREFIX=$1
fi

download(){
  cache_dir=$1
  url=$2
  path=$(echo $url | sed 's/.*:\/\/\(.*\)$/\1/g')
  if [[ -f $cache_dir/$path ]]; then
    return
  fi
  mkdir -p $(dirname $cache_dir/$path)
  if [[ -z "$ARROW_CACHE_URL_PREFIX" ]]; then
    wget -nv -O $cache_dir/$path $url
  else
    wget -nv -O $cache_dir/$path $ARROW_CACHE_URL_PREFIX/$path
  fi
}
export -f download

awk '{print $2}' arrow/thirdparty.list | xargs -P$(nproc) -I{} bash -c "download $CACHE_DIR {}"

if [[ ! -f arrow/src/cpp/CMakeLists.txt ]]; then
  wget -nv -O $CACHE_DIR/src.tar $ARROW_CACHE_URL_PREFIX/src.tar 2>/dev/null || true
  if [[ -f $CACHE_DIR/src.tar ]] && [[ -s $CACHE_DIR/src.tar ]]; then
    tar -xf $CACHE_DIR/src.tar -C arrow/src
  fi
fi

#!/bin/bash

set -eo pipefail

cp -rf arrow/patches/cpp arrow/src/cpp

if [[ ! -d $CACHE_DIR ]]; then
  CACHE_DIR=arrow/cache
fi
CACHE_DIR=$(readlink -f $CACHE_DIR)

while read line; do
  vname=$(echo "$line" | awk '{print $1}')
  url=$(echo "$line" | awk '{print $2}')
  path=$(echo $url | sed 's/.*:\/\/\(.*\)$/\1/g')
  if [[ -f $CACHE_DIR/$path ]]; then
    echo "export $vname=$CACHE_DIR/$path"
    eval "export $vname=$CACHE_DIR/$path"
  fi
done < arrow/thirdparty.list

ARROW_CPP=$(readlink -f arrow/src/cpp)
ARROW_INSTALL=$(readlink -f ${ARROW_INSTALL})
cd ${ARROW_BUILD}

cmake \
-E env CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}" \
cmake \
-DCMAKE_INSTALL_PREFIX:PATH=${ARROW_INSTALL} \
-DARROW_BUILD_SHARED=OFF \
-DARROW_BUILD_STATIC=ON \
-DARROW_SIMD_LEVEL=${SIMD_LEVEL} \
-DARROW_DATASET=ON \
-DARROW_FILESYSTEM=ON \
-DARROW_IPC=ON \
-DARROW_COMPUTE=ON \
-DARROW_CUDA=OFF \
-DARROW_CSV=ON \
-DARROW_JSON=ON \
-DARROW_PARQUET=ON \
-DARROW_WITH_SNAPPY=ON \
-DARROW_WITH_ZSTD=ON \
-DARROW_TENSORFLOW=OFF \
-DARROW_HDFS=${WITH_ARROW_HDFS} \
-DARROW_S3=${WITH_ARROW_S3} \
${ARROW_CPP}

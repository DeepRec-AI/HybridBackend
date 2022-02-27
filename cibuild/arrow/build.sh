#!/bin/bash

set -eo pipefail

HERE=$(dirname $0)
ARROW_DIST=$1

if [[ -z "$ARROW_USE_CXX11_ABI" ]]; then
  export ARROW_USE_CXX11_ABI=0
fi
if [[ -z "$ARROW_SIMD_LEVEL" ]]; then
  export ARROW_SIMD_LEVEL=AVX2
fi
if [[ -z "$ARROW_HDFS" ]]; then
  export ARROW_HDFS=ON
fi
if [[ -z "$ARROW_S3" ]]; then
  export ARROW_S3=ON
fi

cd ${HERE}

mkdir src
SRCTGZ=https://github.com/apache/arrow/archive/refs/tags/apache-arrow-5.0.0.tar.gz
wget -nv ${SRCTGZ} -O /tmp/arrow.tar.gz
tar -xzf /tmp/arrow.tar.gz --strip-components 1 -C src/
cp -rf ./patches/cpp src/

mkdir build
cd build/

if [[ -z "$ARROW_DIST" ]]; then
  mkdir -p ../dist
  export ARROW_DIST=../dist
fi

OS=$(uname -s)
if [[ "${OS}" == "Darwin" ]]; then
ARROW_OSX_TARGET=$(sw_vers -productVersion)
cmake \
-E env CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ARROW_USE_CXX11_ABI} -mmacosx-version-min=${ARROW_OSX_TARGET}" \
cmake \
-DCMAKE_INSTALL_PREFIX:PATH=${ARROW_DIST} \
-DCMAKE_OSX_DEPLOYMENT_TARGET="${ARROW_OSX_TARGET}" \
-DCMAKE_C_FLAGS="-Wno-error=option-ignored" \
-DCMAKE_CXX_FLAGS="-Wno-error=option-ignored" \
-DARROW_BUILD_SHARED=OFF \
-DARROW_BUILD_STATIC=ON \
-DARROW_SIMD_LEVEL=${ARROW_SIMD_LEVEL} \
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
-DARROW_HDFS=${ARROW_HDFS} \
-DARROW_S3=${ARROW_S3} \
../src/cpp
else
cmake \
-E env CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ARROW_USE_CXX11_ABI}" \
cmake \
-DCMAKE_INSTALL_PREFIX:PATH=${ARROW_DIST} \
-DARROW_BUILD_SHARED=OFF \
-DARROW_BUILD_STATIC=ON \
-DARROW_SIMD_LEVEL=${ARROW_SIMD_LEVEL} \
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
-DARROW_HDFS=${ARROW_HDFS} \
-DARROW_S3=${ARROW_S3} \
../src/cpp
fi

make install

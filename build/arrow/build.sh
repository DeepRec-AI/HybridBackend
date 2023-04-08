#!/bin/bash

set -eo pipefail

HERE=$(dirname $0)
ARROW_DIST=$1

export MAKEFLAGS=-j$(nproc)

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

mkdir -p lz4
LZ4TGZ=https://github.com/lz4/lz4/archive/refs/tags/v1.9.4.tar.gz
wget --no-check-certificate -nv ${LZ4TGZ} -O /tmp/lz4.tar.gz
tar -xzf /tmp/lz4.tar.gz --strip-components 1 -C lz4/
cd lz4 && CFLAGS='-O3 -fPIC' make -j8 && cd ..

mkdir -p arrow
SRCTGZ=https://github.com/apache/arrow/archive/refs/tags/apache-arrow-9.0.0.tar.gz
wget --no-check-certificate -nv ${SRCTGZ} -O /tmp/arrow.tar.gz
tar -xzf /tmp/arrow.tar.gz --strip-components 1 -C arrow/

sed -i \
's/ARROW_BOOST_BUILD_SHA256_CHECKSUM=267e04a7c0bfe85daf796dedc789c3a27a76707e1c968f0a2a87bb96331e2b61/ARROW_BOOST_BUILD_SHA256_CHECKSUM=aeb26f80e80945e82ee93e5939baebdca47b9dee80a07d3144be1e1a6a66dd6a/g' \
arrow/cpp/thirdparty/versions.txt

mkdir -p build
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
-DARROW_ORC=ON \
-DARROW_WITH_SNAPPY=ON \
-DARROW_WITH_ZSTD=ON \
-DARROW_WITH_LZ4=ON \
-DARROW_TENSORFLOW=OFF \
-DARROW_HDFS=${ARROW_HDFS} \
-DARROW_S3=${ARROW_S3} \
../arrow/cpp
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
-DARROW_ORC=ON \
-DARROW_WITH_SNAPPY=ON \
-DARROW_WITH_ZSTD=ON \
-DARROW_WITH_LZ4=ON \
-DARROW_TENSORFLOW=OFF \
-DARROW_HDFS=${ARROW_HDFS} \
-DARROW_S3=${ARROW_S3} \
../arrow/cpp
fi

set -x
cp -rf ../patches/cpp ../arrow/
make install

cp -rf ../lz4/lib/liblz4.a* ${ARROW_DIST}/lib/
cp -rf ../lz4/lib/liblz4.so* ${ARROW_DIST}/lib/

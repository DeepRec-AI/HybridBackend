#!/bin/bash

set -eo pipefail

HERE=$(dirname $0)
SPARSEHASH_DIST=$1

SRCTGZ=https://github.com/sparsehash/sparsehash-c11/archive/refs/tags/v2.11.1.tar.gz
wget --no-check-certificate -nv ${SRCTGZ} -O /tmp/sparsehash.tar.gz

cd ${HERE}
if [[ -z "$SPARSEHASH_DIST" ]]; then
  mkdir -p ./dist
  export SPARSEHASH_DIST=./dist
fi
mkdir -p ${SPARSEHASH_DIST}/include
tar -xzf /tmp/sparsehash.tar.gz --strip-components 1 -C ${SPARSEHASH_DIST}/include

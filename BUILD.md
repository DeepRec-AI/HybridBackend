# Build from source

## 1. Linux

### 1.1 Use prebuilt developer docker

[Docker is the future](https://docs.docker.com/engine/install/)

```bash
# Environment variable `DOCKER_RUN_IMAGE` can be used to replace
# developer docker.
cibuild/run make -j$(nproc)
```

### 1.2 Build from scratch

Requirements:

- Ubuntu 18.04 or later (64-bit)
- Python 3.6 or later
- Pip 19.0 or later
- TensorFlow 1.15 or TensorFlow 1.14
- For GPU support, CUDA SDK 11.3 or later is required

Build & install arrow:

```bash
cd cibuild/arrow/
ARROW_USE_CXX11_ABI=${HYBRIDBACKEND_USE_CXX11_ABI} \
ARROW_HDFS=ON \
ARROW_S3=ON \
./build.sh
```

Build & install sparsehash:

```bash
cd cibuild/sparsehash
./build.sh
```

Install TensorFlow and other requirements, see
[Dockerfiles](cibuild/dockerfiles/) for more detail.

Configure & build:

```bash
# Build GPU releated functions.
export HYBRIDBACKEND_WITH_CUDA=ON
# For TensorFlow 1.15, zero-copy is supported.
export HYBRIDBACKEND_WITH_ARROW_ZEROCOPY=ON
# Use below command to verify C++ ABI of installed TensorFlow.
python -c 'import tensorflow as tf; print(tf.sysconfig.get_compile_flags())'
# Must be consistent with installed TensorFlow.
export HYBRIDBACKEND_USE_CXX11_ABI=0

make -j$(nproc)
```

## 2. macOS

Requirements:

- macOS 11.0 or later (x86 64-bit)
- Python 3.7 or later
- Pip 19.0 or later
- Tebnsorflow 1.15 or TensorFlow 1.14
- Other libraries installed by [brew](https://brew.sh/)

Build & install arrow:

```bash
cd cibuild/arrow/
ARROW_USE_CXX11_ABI=${HYBRIDBACKEND_USE_CXX11_ABI} \
ARROW_HDFS=ON \
ARROW_S3=ON \
./build.sh
```

Build & install sparsehash:

```bash
cd cibuild/sparsehash
./build.sh
```

Install TensorFlow and other requirements:

```bash
brew install wget python@3.7 openssl@1.1 utf8proc zstd snappy re2 thrift zlib
brew uninstall grpc abseil || true
export PATH="/usr/local/opt/python@3.7/bin:$PATH"
export LDFLAGS="-L/usr/local/opt/python@3.7/lib"
export PKG_CONFIG_PATH="/usr/local/opt/python@3.7/lib/pkgconfig"

pip3.7 install -i https://mirrors.aliyun.com/pypi/simple/ \
    tensorflow==1.14 \
    "pybind11[global]"
```

Configure & build:

```bash
# Only build CPU releated functions.
export HYBRIDBACKEND_WITH_CUDA=OFF
# For TensorFlow 1.14, zero-copy is not supported.
export HYBRIDBACKEND_WITH_ARROW_ZEROCOPY=OFF
# Use below command to verify C++ ABI of installed TensorFlow.
python -c 'import tensorflow as tf; print(tf.sysconfig.get_compile_flags())'
# Must be consistent with installed TensorFlow.
export HYBRIDBACKEND_USE_CXX11_ABI=0

# Set path of thridparty libraries.
export PYTHON=python3.7
export PYTHON_HOME=/usr/local/opt/python@3.7/Frameworks/Python.framework/Versions/Current
export PYTHON_IMPL=python3.7
export PYTHON_IMPL_FLAG=m
export SSL_HOME=/usr/local/opt/openssl@1.1
export RE2_HOME=/usr/local/opt/re2
export THRIFT_HOME=/usr/local/opt/thrift
export UTF8PROC_HOME=/usr/local/opt/utf8proc
export SNAPPY_HOME=/usr/local/opt/snappy
export ZSTD_HOME=/usr/local/opt/zstd
export ZLIB_HOME=/usr/local/opt/zlib

make -j$(nproc)
```

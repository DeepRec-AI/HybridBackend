# Build from source

## 1. Linux

### 1.1 Use prebuilt container image for developers

[Docker is the future](https://docs.docker.com/engine/install/)

```bash
env/run make -j$(nproc)
```

### 1.2 Use customized container image for developers

Configure container repository and tag:

```bash
export REPO=myhost/myns/myimage
export TAG=deeprec-py3.6-cu114-ubuntu18.04
```

Build and push customized developer image:

```bash
tools/build-developer-container \
--build-arg TF_REPO=https://github.com/alibaba/DeepRec.git \
--build-arg TF_COMMIT=b73e41b8399038373da0f94d204673c911c4dbc1
docker push myhost/myns/myimage:developer-deeprec-py3.6-cu114-ubuntu18.04
```

Build HybridBackend on customized developer image, or build and push customized
image from counterpart developer image:

```bash
env/run make -j$(nproc)
```

Or

```bash
VERSION=xxx \
tools/distbuild
docker push myhost/myns/myimage:xxx-deeprec-py3.6-cu114-ubuntu18.04
```

**NOTE**

`docker-ce >= 20` is required for BuildKit support.

For better debugability:

1. Edit `/etc/systemd/system/multi-user.target.wants/docker.service`
2. Under the `[Service]` tag, put those lines:

```
Environment="BUILDKIT_STEP_LOG_MAX_SIZE=1000000000"
Environment="BUILDKIT_STEP_LOG_MAX_SPEED=10000000"
```

3. Then restart docker daemon:

```bash
systemctl daemon-reload
systemctl restart docker.service
```

### 1.3 Build from scratch

Requirements:

- Ubuntu 18.04 or later (64-bit)
- Python 3.6 or later
- Pip 19.0 or later
- TensorFlow 1.15 or TensorFlow 1.14
- For GPU support, CUDA SDK 11.3 or later is required

Build & install arrow:

```bash
cd env/arrow/
ARROW_USE_CXX11_ABI=${HYBRIDBACKEND_USE_CXX11_ABI} \
ARROW_HDFS=ON \
ARROW_S3=ON \
./build.sh
```

Build & install sparsehash:

```bash
cd env/sparsehash
./build.sh
```

Install TensorFlow and other requirements, see
[Dockerfiles](env/dockerfiles/) for more detail.

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
cd env/arrow/
ARROW_USE_CXX11_ABI=${HYBRIDBACKEND_USE_CXX11_ABI} \
ARROW_HDFS=ON \
ARROW_S3=ON \
./build.sh
```

Build & install sparsehash:

```bash
cd env/sparsehash
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

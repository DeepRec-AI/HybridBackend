# HybridBackend

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![cibuild: cpu](https://github.com/alibaba/HybridBackend/actions/workflows/cpu-cibuild.yaml/badge.svg?branch=main&event=push)](https://github.com/alibaba/HybridBackend/actions/workflows/cpu-cibuild.yaml)
[![Documentation Status](https://readthedocs.org/projects/hybridbackend/badge/?version=latest)](https://hybridbackend.readthedocs.io/en/latest/?badge=latest)

## Introduction

HybridBackend is a training framework for deep recommenders which bridges the
gap between evolving cloud infrastructure and complex training process. See
[documentation](https://hybridbackend.readthedocs.io/en/latest/) for more
information.

![bridging](images/bridging_the_gap.png)

## Requirements

For Linux/macOS installation:

- Ubuntu 18.04 or later (64-bit)
- Python 3.6 or later
- Pip 19.0 or later
- TensorFlow 1.15 or TensorFlow 1.14
- For GPU support, CUDA SDK 11.3 or later is required
- [Docker is the future](https://docs.docker.com/engine/install/)

For docker-phobes using macOS:

- macOS 11.0 or later (x86 64-bit)
- Python 3.7 or later
- Pip 19.0 or later
- Tebnsorflow 1.15 or TensorFlow 1.14
- Other libraries installed by [brew](https://brew.sh/)

## Install

For TensorFlow 1.15 CPU version:

```bash
pip install hybridbackend-cpu
```

For TensorFlow 1.14 CPU version:

```bash
pip install hybridbackend-cpu-legacy
```

For GPU support:

[PAI DLC](https://www.aliyun.com/activity/bigdata/pai-dlc) docker images are
prefered to use.

## Build from source on Linux/macOS w/ docker

- Fetch source from git and sync submodules.

```bash
git submodule sync
git submodule update --init
```

- Step into developer docker.

```bash
# Environment variable `DOCKER_RUN_IMAGE` can be used to replace
# developer docker.
cibuild/run
```

- Configure.

```bash
# Only build CPU releated functions.
export HYBRIDBACKEND_WITH_CUDA=OFF
# For TensorFlow 1.14, zero-copy is not supported.
export HYBRIDBACKEND_WITH_ARROW_ZEROCOPY=OFF
# Use below command to verify C++ ABI of installed TensorFlow.
python -c 'import tensorflow as tf; print(tf.sysconfig.get_compile_flags())'
# Must be consistent with installed TensorFlow.
export HYBRIDBACKEND_USE_CXX11_ABI=0
```

- Build.

```bash
cibuild/run make -j8
```

## Build from source on Linux w/o docker

- Fetch source from git and sync submodules.

```bash
git submodule sync
git submodule update --init
```

- Install TensorFlow and other requirements.

See [Dockerfiles](cibuild/dockerfiles/).

- Configure.

```bash
# Only build CPU releated functions.
export HYBRIDBACKEND_WITH_CUDA=OFF
# For TensorFlow 1.15, zero-copy is supported.
export HYBRIDBACKEND_WITH_ARROW_ZEROCOPY=ON
# Use below command to verify C++ ABI of installed TensorFlow.
python -c 'import tensorflow as tf; print(tf.sysconfig.get_compile_flags())'
# Must be consistent with installed TensorFlow.
export HYBRIDBACKEND_USE_CXX11_ABI=0
```

- Build.

```bash
make -j8
```

## Build from source on macOS w/o docker

- Fetch source from git and sync submodules.

```bash
git submodule sync
git submodule update --init
```

- Install TensorFlow and other requirements.

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

- Configure.

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
```

- Build.

```bash
make -j8
```

## Contributing

We appreciate all contributions to improve HybridBackend. Please see
[Contributing Guide](CONTRIBUTING.md) for more details.

## Community

If you are intrested in adoption of HybridBackend in your organization, you can
add your organization name to our [list of adopters](ADOPTERS.md) by submitting
a pull request. We will discuss new feature requirements with you in advance.

Further more, if you would like to share your experiences with others, you are
welcome to contact us in DingTalk:

<img src="images/hbcommunity.png" alt="community" width="200"/>

## License

HybridBackend is licensed under the [Apache 2.0 License](LICENSE).

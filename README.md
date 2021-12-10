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

## Installation

Install latest CPU version for TensorFlow 1.15:

```bash
pip install hybridbackend-cpu
```

Install latest CPU version for TensorFlow 1.14:

```bash
pip install hybridbackend-cpu-legacy
```

Note:

You might need to upgrade pip before above installations:

```bash
pip install -U pip
```

## Contributing

We appreciate all contributions to improve HybridBackend. Please see
[Contributing Guide](CONTRIBUTING.md) for more details.

## License

HybridBackend is licensed under the [Apache 2.0 License](LICENSE).

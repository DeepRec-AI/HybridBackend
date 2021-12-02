# HybridBackend

![Tensorflow 1.15 CPU CI Build Badge](https://github.com/alibaba/HybridBackend/actions/workflows/.github/workflows/tensorflow1.15-py3.6-cibuild.yaml/badge.svg)
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

We appreciate all contributions to improve HybridBackend. Please follow below
steps to contribute:

**1. Clone the repository and checkout a new branch.**

```bash
git clone <git_repo_addr>
git pull -r
git checkout -b features/my_feature
```

**2. Commit changes, check code style and test.**

```bash
git commit
cibuild/run cibuild/format
cibuild/run cibuild/lint
cibuild/run make -j8
cibuild/run make test
```

**3. Create pull request for code review.**

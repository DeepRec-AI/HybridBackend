# HybridBackend

[![cibuild](https://github.com/alibaba/HybridBackend/actions/workflows/cibuild.yaml/badge.svg?branch=main&event=push)](https://github.com/alibaba/HybridBackend/actions/workflows/cibuild.yaml)
[![readthedocs](https://readthedocs.org/projects/hybridbackend/badge/?version=latest)](https://hybridbackend.readthedocs.io/en/latest/?badge=latest)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![license](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

HybridBackend is a high-performance framework for training wide-and-deep
recommender systems on heterogeneous cluster.

## Features

- Memory-efficient loading of categorical data

- GPU-efficient orchestration of embedding layers

- Communication-efficient training and evaluation at scale

- Easy to use with existing AI workflows

## Usage

A minimal example:

```python
import tensorflow as tf
import hybridbackend.tensorflow as hb

ds = hb.data.ParquetDataset(filenames, batch_size=batch_size)
ds = ds.apply(hb.data.parse())
# ...

with tf.device('/gpu:0'):
  embs = tf.nn.embedding_lookup_sparse(weights, input_ids)
  # ...
```

Please see [documentation](https://hybridbackend.readthedocs.io/en/latest/) for
more information.

## Install

### Method 1: Pull container images from [PAI DLC](https://www.aliyun.com/activity/bigdata/pai-dlc)

`docker pull registry.cn-shanghai.aliyuncs.com/pai-dlc/hybridbackend:{TAG}`

`{TAG}` | TensorFlow | Python  | CUDA | OS | Columnar Data Loading | Embedding Orchestration | Hybrid Parallelism
------- | ---------- | ------- | ---- | ----- | ------------ | ----------------------- | ------------------
`0.6-tf1.15-py3.8-cu114-ubuntu20.04` | 1.15 | 3.8 | 11.4 | Ubuntu 20.04 | &check; | &check; | &check;

### Method 2: Install from PyPI

`pip install {PACKAGE}`

`{PACKAGE}` | TensorFlow | Python  | CUDA | GLIBC | Columnar Data Loading | Embedding Orchestration | Hybrid Parallelism
----------- | ---------- | ------- | ---- | ----- | ------------ | ----------------------- | ------------------
[hybridbackend-tf115-cu114](https://pypi.org/project/hybridbackend-tf115-cu114/) `*` | 1.15 | 3.8 | 11.4 | >=2.31 | &check; | &check; | &check;
[hybridbackend-tf115-cu100](https://pypi.org/project/hybridbackend-tf115-cu100/) | 1.15 | 3.6 | 10.0 | >=2.27 | &check; | &check; | &cross;
[hybridbackend-tf115-cpu](https://pypi.org/project/hybridbackend-tf115-cpu/) | 1.15 | 3.6 | - | >=2.24 | &check; | &cross; | &cross;

> `*` [nvidia-pyindex](https://pypi.org/project/nvidia-pyindex/) must be installed first

### Method 3: Build from source

See [Building Instructions](https://github.com/alibaba/HybridBackend/blob/main/BUILD.md).

## License

HybridBackend is licensed under the [Apache 2.0 License](LICENSE).

## Community

- Please see [Contributing Guide](https://github.com/alibaba/HybridBackend/blob/main/CONTRIBUTING.md)
before your first contribution.

- Please [register as an adopter](https://github.com/alibaba/HybridBackend/blob/main/ADOPTERS.md)
if your organization is interested in adoption. We will discuss
[RoadMap](https://github.com/alibaba/HybridBackend/blob/main/ROADMAP.md) with
registered adopters in advance.

- Please cite [HybridBackend](https://ieeexplore.ieee.org/document/9835450) in your publications if it helps:

  ```text
  @inproceedings{zhang2022picasso,
    title={PICASSO: Unleashing the Potential of GPU-centric Training for Wide-and-deep Recommender Systems},
    author={Zhang, Yuanxing and Chen, Langshi and Yang, Siran and Yuan, Man and Yi, Huimin and Zhang, Jie and Wang, Jiamang and Dong, Jianbo and Xu, Yunlong and Song, Yue and others},
    booktitle={2022 IEEE 38th International Conference on Data Engineering (ICDE)},
    year={2022},
    organization={IEEE}
  }
  ```

## Contact Us

If you would like to share your experiences with others, you are welcome to
contact us in DingTalk:

[<img src="https://github.com/alibaba/HybridBackend/raw/main/images/dingtalk.png" alt="dingtalk" width="200"/>](https://h5.dingtalk.com/circle/healthCheckin.html?dtaction=os&corpId=ding14f3e2ea4b79994cadf6428847a62d4a&51951ad=a84b419&cbdbhh=qwertyuiop)

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

ds = hb.data.Dataset.from_parquet(filenames)
ds = ds.batch(batch_size)
# ...

with tf.device('/gpu:0'):
  embs = tf.nn.embedding_lookup_sparse(weights, input_ids)
  # ...
```

Please see [documentation](https://hybridbackend.readthedocs.io/en/latest/) for
more information.

## Install

### Method 1: Install from PyPI

`pip install {PACKAGE}`

| `{PACKAGE}`                                                                             | Dependency                                                              | Python | CUDA | GLIBC  | Data Opt. | Embedding Opt. | Parallelism Opt. |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------ | ---- | ------ | --------- | -------------- | ---------------- |
| [hybridbackend-tf115-cu121](https://pypi.org/project/hybridbackend-tf115-cu121/)             | [TensorFlow 1.15](https://github.com/NVIDIA/tensorflow)  | 3.8    | 12.1 | >=2.31 | &check;   | &check;        | &check;          |
| [hybridbackend-tf115-cu100](https://pypi.org/project/hybridbackend-tf115-cu100/)             | [TensorFlow 1.15](https://github.com/tensorflow/tensorflow/tree/r1.15)     | 3.6    | 10.0 | >=2.27 | &check;   | &check;        | &cross;          |
| [hybridbackend-tf115-cpu](https://pypi.org/project/hybridbackend-tf115-cpu/)                 | [TensorFlow 1.15](https://github.com/tensorflow/tensorflow/tree/r1.15)     | 3.6    | -    | >=2.24 | &check;   | &cross;        | &cross;          |

### Method 2: Build from source

See [Building Instructions](https://github.com/alibaba/HybridBackend/blob/main/BUILD.md).

We also provide built docker images for latest [DeepRec](https://github.com/alibaba/DeepRec): 
`registry.cn-shanghai.aliyuncs.com/pai-dlc/hybridbackend:1.0.0-deeprec-py3.6-cu114-ubuntu18.04`

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

[![dingtalk](https://github.com/alibaba/HybridBackend/raw/main/docs/images/dingtalk.png)](https://qr.dingtalk.com/action/joingroup?code=v1,k1,VouhbeuTwXYEgaLzSOE8o6VF2kTHVJ8lw5h93WbZW8o=&_dt_no_comment=1&origin=11)
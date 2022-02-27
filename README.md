# HybridBackend

[![cibuild: cpu](https://github.com/alibaba/HybridBackend/actions/workflows/cpu-cibuild.yaml/badge.svg?branch=main&event=push)](https://github.com/alibaba/HybridBackend/actions/workflows/cpu-cibuild.yaml)
[![readthedocs](https://readthedocs.org/projects/hybridbackend/badge/?version=latest)](https://hybridbackend.readthedocs.io/en/latest/?badge=latest)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![license](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

HybridBackend is a high-performance framework for training wide-and-deep
recommender systems on heterogeneous cluster.

## Features

- Memory-efficient loading of categorical data

## Install

### Using pip packages

GLIBC    | CUDA | Python | Tensorflow      | Command
-------- | ---- | ------ | --------------- | ------------
`>= 2.4` | -    | 3.6    | `>=1.15, < 2.0` | `pip install hybridbackend-cpu`
`>= 2.4` | -    | 3.6    | `>=1.14, < 1.15` | `pip install hybridbackend-cpu-legacy`

### Build from source

See [Building Instructions](https://github.com/alibaba/HybridBackend/blob/main/BUILD.md).

## Usage

Please see [documentation](https://hybridbackend.readthedocs.io/en/latest/) for
more information.

## License

HybridBackend is licensed under the [Apache 2.0 License](LICENSE).

## Community

- Please see [Contributing Guide](https://github.com/alibaba/HybridBackend/blob/main/CONTRIBUTING.md)
before your first contribution.

- Please [register as an adopter](https://github.com/alibaba/HybridBackend/blob/main/ADOPTERS.md)
if your organization is interested in adoption. We will discuss new feature
requirements with registered adopters in advance.

- Please cite HybridBackend in your publications if it helps:

  ```text
  @article{zhang2022picasso,
    title={PICASSO: Unleashing the Potential of GPU-centric Training for Wide-and-deep Recommender Systems},
    author={Zhang, Yuanxing and Chen, Langshi and Yang, Siran and Yuan, Man and Yi, Huimin and Zhang, Jie and Wang, Jiamang and Dong, Jianbo and Xu, Yunlong and Song, Yue and others},
    journal={arXiv preprint arXiv:2204.04903},
    year={2022}
  }
  ```

## Contact Us

If you would like to share your experiences with others, you are welcome to
contact us in DingTalk:

[<img src="https://github.com/alibaba/HybridBackend/raw/main/images/dingtalk.png" alt="dingtalk" width="200"/>](https://h5.dingtalk.com/circle/healthCheckin.html?dtaction=os&corpId=ding14f3e2ea4b79994cadf6428847a62d4a&51951ad=a84b419&cbdbhh=qwertyuiop)

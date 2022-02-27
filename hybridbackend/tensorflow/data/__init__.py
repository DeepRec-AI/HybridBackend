# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

r'''Input pipelines.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hybridbackend.tensorflow.data.adapter import make_one_shot_iterator
from hybridbackend.tensorflow.data.adapter import make_initializable_iterator
from hybridbackend.tensorflow.data.dataframe import DataFrame
from hybridbackend.tensorflow.data.dataframe import to_sparse
from hybridbackend.tensorflow.data.dataframe import unbatch_and_to_sparse
from hybridbackend.tensorflow.data.parquet_dataset import ParquetDataset
from hybridbackend.tensorflow.data.parquet_dataset import read_parquet
from hybridbackend.tensorflow.data.rebatch_dataset import RebatchDataset
from hybridbackend.tensorflow.data.rebatch_dataset import rebatch

# HybridBackend operators must be loaded before TensorFlow operators to
# make AWS SDK implementation correct.
from hybridbackend.tensorflow.data.dataset_ops import make_iterator

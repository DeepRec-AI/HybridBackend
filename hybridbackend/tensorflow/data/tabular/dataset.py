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

r'''Dataset that reads tabular data.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.deprecation import deprecated

# pylint: disable=ungrouped-imports
try:
  from hybridbackend.tensorflow.data.tabular.dataset_v2 import \
    TabularDatasetV2 as Dataset
  Dataset.__module__ = __name__
  Dataset.__name__ = 'TabularDataset'
except ImportError:
  from hybridbackend.tensorflow.data.tabular.dataset_v1 import \
    TabularDatasetV1 as Dataset
  Dataset.__module__ = __name__
  Dataset.__name__ = 'TabularDataset'

try:
  from tensorflow.python.data.ops.dataset_ops import DatasetV2 as _dataset  # pylint: disable=unused-import, line-too-long # noqa: F401

  from hybridbackend.tensorflow.data.tabular.dataset_v2 import \
    ParquetDatasetV2 as ParquetDataset
  ParquetDataset.__module__ = __name__
  ParquetDataset.__name__ = 'ParquetDataset'
except ImportError:
  from hybridbackend.tensorflow.data.tabular.dataset_v1 import \
    ParquetDatasetV1 as ParquetDataset
  ParquetDataset.__module__ = __name__
  ParquetDataset.__name__ = 'ParquetDataset'
# pylint: enable=ungrouped-imports


@deprecated(None, 'Prefer hb.data.Dataset.from_parquet instead.')
def read_parquet(
    batch_size,
    fields=None,
    partition_count=1,
    partition_index=0,
    drop_remainder=False,
    num_parallel_reads=None,
    num_sequential_reads=1):
  r'''Create a `ParquetDataset` from filenames dataset.

    Args:
      batch_size: Maxium number of samples in an output batch.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      drop_remainder: (Optional.) If True, only keep batches with exactly
        `batch_size` samples.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_sequential_reads: (Optional.) A `tf.int64` scalar representing the
        number of batches to read in sequential. Defaults to 1.
    '''
  def _apply_fn(filenames):
    return ParquetDataset(
      filenames,
      batch_size=batch_size,
      fields=fields,
      partition_count=partition_count,
      partition_index=partition_index,
      drop_remainder=drop_remainder,
      num_parallel_reads=num_parallel_reads,
      num_sequential_reads=num_sequential_reads)
  return _apply_fn

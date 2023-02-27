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

r'''Dataset that resizes batches of DataFrame values.

This class is compatible with TensorFlow 1.15.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import nest

from hybridbackend.tensorflow.common import oplib as _ops
from hybridbackend.tensorflow.data.dataframe import input_fields


class RebatchDatasetV2(dataset_ops.DatasetV2):
  r'''A dataset that adjusts batches.
  '''
  def __init__(
      self, input_dataset,
      batch_size,
      min_batch_size=None,
      fields=None,
      drop_remainder=False,
      num_parallel_scans=1):
    r'''Create a `RebatchDatasetV2`.

    Args:
      input_dataset: A dataset outputs batches.
      batch_size: Maxium number of samples in an output batch.
      min_batch_size: (Optional.) Minimum number of samples in a non-final
        batch. Same to `batch_size` by default.
      fields: (Optional.) List of DataFrame fields. Fetched from `input_dataset`
        by default.
      drop_remainder: (Optional.) If True, smaller final batch is dropped.
        `False` by default.
      num_parallel_scans: (Optional.) Number of concurrent scans against fields
        of input dataset.
    '''
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
      batch_size,
      dtype=dtypes.int64,
      name='batch_size')
    if min_batch_size is None:
      min_batch_size = batch_size
    self._min_batch_size = ops.convert_to_tensor(
      min_batch_size,
      dtype=dtypes.int64,
      name='min_batch_size')
    self._fields = input_fields(input_dataset, fields)
    self._drop_remainder = drop_remainder
    if num_parallel_scans == dataset_ops.AUTOTUNE:
      num_parallel_scans = len(self._fields)
    self._num_parallel_scans = num_parallel_scans
    self._impl = _ops.hb_rebatch_tabular_dataset(
      self._input_dataset._variant_tensor,  # pylint: disable=protected-access
      self._batch_size,
      self._min_batch_size,
      field_ids=nest.flatten({
        f.name: f.map(lambda _, j=idx: j)
        for idx, f in enumerate(self._fields)}),
      field_ragged_indices=nest.flatten(
        {f.name: f.ragged_indices for f in self._fields}),
      drop_remainder=self._drop_remainder,
      num_parallel_scans=self._num_parallel_scans)
    super().__init__(self._impl)

  @property
  def fields(self):
    return self._fields

  @property
  def drop_remainder(self):
    return self._drop_remainder

  @property
  def num_parallel_scans(self):
    return self._num_parallel_scans

  def _inputs(self):
    return [self._input_dataset]

  @property
  def element_spec(self):
    return self._input_dataset.element_spec  # pylint: disable=protected-access

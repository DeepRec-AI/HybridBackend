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

This class is compatible with TensorFlow 1.12.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest

from hybridbackend.tensorflow.common import oplib as _ops


class RebatchDatasetV1(dataset_ops.Dataset):
  r'''A dataset that adjusts batches.
  '''
  def __init__(
      self, input_dataset, fields, batch_size,
      drop_remainder=False,
      shuffle_buffer_size=None,
      shuffle_seed=None,
      reshuffle_each_iteration=None):
    r'''Create a `RebatchDatasetV1`.

    Args:
      input_dataset: A dataset outputs batches.
      fields: List of DataFrame fields. Fetched from `input_dataset`
        by default.
      batch_size: Maxium number of samples in an output batch.
      drop_remainder: (Optional.) If True, smaller final batch is dropped.
        `False` by default.
      shuffle_buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the
        number of elements from this dataset from which the new
        dataset will sample.
      shuffle_seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing
        the random seed that will be used to create the distribution. See
        @{tf.set_random_seed} for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)
    '''
    self._input_dataset = input_dataset
    self._fields = fields
    self._batch_size = ops.convert_to_tensor(
      batch_size,
      dtype=dtypes.int64,
      name='batch_size')
    self._drop_remainder = drop_remainder
    self._shuffle_buffer_size = ops.convert_to_tensor(
      shuffle_buffer_size or 0, dtype=dtypes.int64, name='buffer_size')
    self._seed, self._seed2 = random_seed.get_seed(shuffle_seed)
    self._reshuffle_each_iteration = reshuffle_each_iteration or True
    super().__init__()

  @property
  def fields(self):
    return self._fields

  @property
  def drop_remainder(self):
    return self._drop_remainder

  def _as_variant_tensor(self):
    return _ops.hb_rebatch_tabular_dataset_v2(
      self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
      self._batch_size,
      self._shuffle_buffer_size,
      self._seed,
      self._seed2,
      field_ids=nest.flatten({
        f.name: f.map(lambda _, j=idx: j)
        for idx, f in enumerate(self._fields)}),
      field_ragged_indices=nest.flatten(
        {f.name: f.ragged_indices for f in self._fields}),
      drop_remainder=self._drop_remainder,
      reshuffle_each_iteration=self._reshuffle_each_iteration,
      output_types=self.output_types,
      output_shapes=self.output_shapes)

  def _inputs(self):
    return [self._input_dataset]

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_types(self):
    return self._input_dataset.output_types

  @property
  def output_shapes(self):
    input_shapes = self._input_dataset.output_shapes
    dim0_shape = tensor_shape.TensorShape([
      tensor_util.constant_value(self._batch_size)
      if self._drop_remainder
      else None])
    return nest.pack_sequence_as(
      input_shapes,
      [
        dim0_shape.concatenate(s[1:])
        for s in nest.flatten(self._input_dataset.output_shapes)])

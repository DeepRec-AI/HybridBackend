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

r'''Dataset that resizes batches of tensors.

This class is compatible with TensorFlow 1.12.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest

from hybridbackend.tensorflow.common import oplib as _ops
from hybridbackend.tensorflow.framework.ops import TensorKinds


class _RectifyDatasetV1(dataset_ops.Dataset):
  r'''A dataset that adjusts batches.
  '''
  def __init__(
      self, input_dataset, output_kinds, batch_size,
      drop_remainder=False,
      shuffle_buffer_size=None,
      shuffle_seed=None,
      reshuffle_each_iteration=None):
    r'''Create a `RectifyDatasetV2`.

    Args:
      input_dataset: A dataset outputs batches.
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
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._shuffle_buffer_size = ops.convert_to_tensor(
      shuffle_buffer_size or 0, dtype=dtypes.int64, name='buffer_size')
    self._seed, self._seed2 = random_seed.get_seed(shuffle_seed)
    self._reshuffle_each_iteration = reshuffle_each_iteration or True
    self._output_kinds = output_kinds

    dim0_shape = tensor_shape.TensorShape([
      tensor_util.constant_value(self._batch_size)
      if self._drop_remainder
      else None])
    self._output_shapes = []
    for idx, kind in enumerate(self._output_kinds):
      input_shape = self._input_dataset.output_shapes[idx]
      if kind == TensorKinds.INDICES:
        self._output_shapes.append(input_shape)
      elif kind == TensorKinds.VALUES:
        self._output_shapes.append(dim0_shape.concatenate(input_shape[1:]))
      elif kind == TensorKinds.DENSE_SHAPE:
        self._output_shapes.append(input_shape)
      else:
        raise ValueError('Unknown tensor kind: ', kind)
    self._output_shapes = tuple(self._output_shapes)

    super().__init__()

  def _inputs(self):
    return [self._input_dataset]

  def _as_variant_tensor(self):
    return _ops.hb_rectify_dataset(
      self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
      self._batch_size,
      self._shuffle_buffer_size,
      self._seed,
      self._seed2,
      drop_remainder=self._drop_remainder,
      reshuffle_each_iteration=self._reshuffle_each_iteration,
      output_kinds=self._output_kinds,
      output_types=self.output_types,
      output_shapes=self.output_shapes)

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_types(self):
    return self._input_dataset.output_types

  @property
  def output_shapes(self):
    r'''Output shapes.
    '''
    return self._output_shapes


class RectifyDatasetV1(dataset_ops.Dataset):
  r'''A dataset that adjusts batches.
  '''
  VERSION = 2001

  def __init__(
      self, input_dataset, batch_size,
      drop_remainder=False,
      shuffle_buffer_size=None,
      shuffle_seed=None,
      reshuffle_each_iteration=None):
    r'''Create a `RectifyDatasetV1`.

    Args:
      input_dataset: A dataset outputs batches.
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
    flat_shapes = input_dataset._flat_shapes  # pylint: disable=protected-access
    if any(s.ndims == 0 for s in flat_shapes):
      raise ValueError('Cannot rectify an input with scalar components.')
    known_batch_dim = tensor_shape.Dimension(None)
    for s in flat_shapes:
      try:
        known_batch_dim = known_batch_dim.merge_with(s[0])
      except ValueError as ex:
        raise ValueError(
          'Cannot rectify an input whose components have different '
          'batch sizes.') from ex

    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
      batch_size,
      dtype=dtypes.int64,
      name='batch_size')
    self._drop_remainder = drop_remainder

    flattened_classes = nest.flatten(input_dataset.output_classes)
    flattened_kinds = []
    for cls in flattened_classes:
      if cls == ops.Tensor:
        flattened_kinds.append(TensorKinds.VALUES)
      elif cls == sparse_tensor.SparseTensor:
        flattened_kinds.append(
          sparse_tensor.SparseTensorValue(
            TensorKinds.INDICES, TensorKinds.VALUES, TensorKinds.DENSE_SHAPE))
      else:
        raise ValueError(
          'hb.data.rectify cannot support input datasets with outputs other '
          'than tensors or sparse tensors')
    normalized_kinds = nest.flatten(flattened_kinds)
    normalized_dataset = input_dataset.map(TensorKinds.normalize)
    rectify_dataset = _RectifyDatasetV1(  # pylint: disable=abstract-class-instantiated
      normalized_dataset, normalized_kinds,
      batch_size, drop_remainder=drop_remainder,
      shuffle_buffer_size=shuffle_buffer_size,
      shuffle_seed=shuffle_seed,
      reshuffle_each_iteration=reshuffle_each_iteration)
    self._impl = rectify_dataset.map(
      lambda *args: TensorKinds.denormalize(
        input_dataset.element_spec, flattened_kinds, args))

    super().__init__()

  @property
  def drop_remainder(self):
    return self._drop_remainder

  def _as_variant_tensor(self):
    return self._impl._as_variant_tensor()  # pylint: disable=protected-access

  def _inputs(self):
    return [self._input_dataset]

  @property
  def output_classes(self):
    return self._impl.output_classes

  @property
  def output_types(self):
    return self._impl.output_types

  @property
  def output_shapes(self):
    return self._impl.output_shapes

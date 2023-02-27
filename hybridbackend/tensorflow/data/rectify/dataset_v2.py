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

This class is compatible with TensorFlow 1.15.
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
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest

from hybridbackend.tensorflow.common import oplib as _ops
from hybridbackend.tensorflow.framework.ops import TensorKinds


class _RectifyDatasetV2(dataset_ops.DatasetV2):
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
    self._element_spec = []
    for idx, kind in enumerate(self._output_kinds):
      spec = self._input_dataset.element_spec[idx]
      if kind == TensorKinds.INDICES:
        self._element_spec.append(spec)
      elif kind == TensorKinds.VALUES:
        self._element_spec.append(
          tensor_spec.TensorSpec(
            dim0_shape.concatenate(spec.shape[1:]), spec.dtype, spec.name))
      elif kind == TensorKinds.DENSE_SHAPE:
        self._element_spec.append(spec)
      else:
        raise ValueError('Unknown tensor kind: ', kind)
    self._element_spec = tuple(self._element_spec)

    super().__init__(
      _ops.hb_rectify_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        batch_size,
        self._shuffle_buffer_size,
        self._seed,
        self._seed2,
        drop_remainder=drop_remainder,
        reshuffle_each_iteration=self._reshuffle_each_iteration,
        output_kinds=output_kinds,
        **self._flat_structure))

  def _inputs(self):
    return [self._input_dataset]

  @property
  def element_spec(self):
    r'''Element Specification.
    '''
    return self._element_spec


class RectifyDatasetV2(dataset_ops.DatasetV2):  # pylint: disable=abstract-method
  r'''A dataset that adjusts batches.
  '''
  VERSION = 2002

  def __init__(
      self, input_dataset, batch_size,
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

    flattened_specs = nest.flatten(input_dataset.element_spec)
    flattened_kinds = []
    for spec in flattened_specs:
      if isinstance(spec, tensor_spec.TensorSpec):
        flattened_kinds.append(TensorKinds.VALUES)
      elif isinstance(spec, sparse_tensor.SparseTensorSpec):
        flattened_kinds.append(
          sparse_tensor.SparseTensorValue(
            TensorKinds.INDICES, TensorKinds.VALUES, TensorKinds.DENSE_SHAPE))
      else:
        raise ValueError(
          'hb.data.rectify cannot support input datasets with outputs other '
          'than tensors or sparse tensors')
    normalized_kinds = nest.flatten(flattened_kinds)
    normalized_dataset = input_dataset.map(TensorKinds.normalize)
    rectify_dataset = _RectifyDatasetV2(  # pylint: disable=abstract-class-instantiated
      normalized_dataset, normalized_kinds,
      batch_size, drop_remainder=drop_remainder,
      shuffle_buffer_size=shuffle_buffer_size,
      shuffle_seed=shuffle_seed,
      reshuffle_each_iteration=reshuffle_each_iteration)
    self._impl = rectify_dataset.map(
      lambda *args: TensorKinds.denormalize(
        input_dataset.element_spec, flattened_kinds, args))
    super().__init__(self._impl._variant_tensor)  # pylint: disable=protected-access

  @property
  def drop_remainder(self):
    return self._drop_remainder

  def _inputs(self):
    return [self._input_dataset]

  @property
  def element_spec(self):
    return self._impl.element_spec  # pylint: disable=protected-access

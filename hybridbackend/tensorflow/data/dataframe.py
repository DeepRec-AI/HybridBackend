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

r'''Data frame releated classes.

See https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html for
more information about data frame concept.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor


class DataFrame(object):  # pylint: disable=useless-object-inheritance
  r'''Tabular data to train in a deep recommender.'''
  class Field(object):  # pylint: disable=useless-object-inheritance
    r'''Definition of a field in a data frame.
    '''
    def __init__(
        self, name, dtype=dtypes.int64, shape=None, ragged_rank=0):
      self._name = name
      try:
        self._dtype = dtypes.as_dtype(dtype)
      except TypeError:
        if dtype == np.object_:
          self._dtype = dtypes.as_dtype(np.object)
        else:
          raise
      self._shape = shape if shape else [None]
      self._ragged_rank = int(ragged_rank)

    @property
    def name(self):
      return self._name

    @property
    def dtype(self):
      return self._dtype

    @property
    def shape(self):
      return self._shape

    @property
    def ragged_rank(self):
      return self._ragged_rank

    @property
    def ragged_indices(self):
      return self.map(lambda i: i)

    @property
    def output_classes(self):
      return self.map(lambda _: ops.Tensor)

    @property
    def output_types(self):
      return self.map(lambda i: self._dtype if i == 0 else dtypes.int32)

    @property
    def output_shapes(self):
      return self.map(lambda _: tensor_shape.TensorShape([None]))

    def __repr__(self):
      return \
          f'{self._name} <dtype={self._dtype} ' + \
          f'ragged_rank={self._ragged_rank} shape={self._shape}>'

    def map(self, func):
      if self._ragged_rank == 0:
        return func(0)
      return DataFrame.Value(
          func(0),
          tuple(func(i+1) for i in range(self._ragged_rank)))

  # pylint: disable=inherit-non-class
  class Value(
      collections.namedtuple(
          'DataFrameValue',
          ['values', 'nested_row_splits'])):
    r'''A structure represents a value in DataFrame.
    '''
    def to_ragged(self, name=None):
      return ragged_tensor.RaggedTensor.from_nested_row_splits(
          self.values, self.nested_row_splits,
          validate=False,
          name=name)

    def to_sparse(self, name=None):
      sparse_value = gen_ragged_conversion_ops.ragged_tensor_to_sparse(
          self.nested_row_splits, self.values, name=name)
      return sparse_tensor.SparseTensor(
          sparse_value.sparse_indices,
          sparse_value.sparse_values,
          sparse_value.sparse_dense_shape)

  @classmethod
  def to_sparse(cls, features):
    r'''Convert DataFrame values to tensors or sparse tensors.
    '''
    if isinstance(features, dict):
      return {f: cls.to_sparse(features[f]) for f in features}
    if isinstance(features, DataFrame.Value):
      if len(features.nested_row_splits) >= 1:
        features = features.to_sparse()
      else:
        features = features.values
      return features
    if isinstance(features, ragged_tensor.RaggedTensor):
      if features.ragged_rank >= 1:
        features = features.to_sparse()
      return features
    if isinstance(features, ops.Tensor):
      return features
    raise ValueError(f'{features} not supported')

  @classmethod
  def unbatch_and_to_sparse(cls, features):
    r'''Unbatch and convert a row of DataFrame to tensors or sparse tensors.
    '''
    if isinstance(features, dict):
      return {
          f: cls.unbatch_and_to_sparse(features[f])
          for f in features}
    if isinstance(features, DataFrame.Value):
      if len(features.nested_row_splits) > 1:
        features = features.to_sparse()
        features = sparse_ops.sparse_reshape(
            features, features.dense_shape[1:])
      elif len(features.nested_row_splits) == 1:
        num_elems = math_ops.cast(
            features.nested_row_splits[0][1], dtype=dtypes.int64)
        indices = math_ops.range(num_elems)
        indices = array_ops.reshape(indices, [-1, 1])
        features = sparse_tensor.SparseTensor(
            indices, features.values, [-1])
      else:
        features = features.values
        features = array_ops.reshape(features, features.shape[1:])
      return features
    if isinstance(features, ragged_tensor.RaggedTensor):
      if features.ragged_rank > 1:
        features = features.to_sparse()
        features = sparse_ops.sparse_reshape(
            features, features.dense_shape[1:])
      elif features.ragged_rank == 1:
        actual_batch_size = math_ops.cast(
            features.row_splits[1], dtype=dtypes.int64)
        indices = math_ops.range(actual_batch_size)
        indices = array_ops.reshape(indices, [-1, 1])
        features = sparse_tensor.SparseTensor(
            indices, features.values, [-1])
      return features
    if isinstance(features, ops.Tensor):
      return array_ops.reshape(features, features.shape[1:])
    raise ValueError(f'{features} not supported for transformation')


def to_sparse(num_parallel_calls=None):
  r'''Convert values to tensors or sparse tensors from input dataset.
  '''
  def _apply_fn(dataset):
    return dataset.map(
        DataFrame.to_sparse,
        num_parallel_calls=num_parallel_calls)
  return _apply_fn


def unbatch_and_to_sparse(num_parallel_calls=None):
  r'''Unbatch and convert a row to tensors or sparse tensors from input dataset.
  '''
  def _apply_fn(dataset):
    return dataset.map(
        DataFrame.unbatch_and_to_sparse,
        num_parallel_calls=num_parallel_calls)
  return _apply_fn

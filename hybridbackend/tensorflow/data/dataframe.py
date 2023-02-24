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
import logging
from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec

try:
  from tensorflow.python.framework.type_spec import BatchableTypeSpec
except ImportError:
  BatchableTypeSpec = object
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
    class Spec(BatchableTypeSpec):
      r'''A TypeSpec for reading from a value of DataFrame.
      '''
      def __init__(self, field):
        r'''Constructs a type specification for a DataFrame.Value.
        '''
        self._field = field

      def value_type(self):
        return ops.Tensor if self._field.ragged_rank == 0 else DataFrame.Value

      def _serialize(self):
        return (
          self._field.name,
          self._field.dtype,
          self._field.ragged_rank,
          self._field.shape)

      @property
      def _component_specs(self):
        r'''Component DataSpecs.
        '''
        if self._field.ragged_rank == 0:
          return tensor_spec.TensorSpec(
            shape=tensor_shape.TensorShape([None]).concatenate(
              self._field.shape),
            dtype=self._field.dtype,
            name=self._field.name)
        return DataFrame.Value(
          tensor_spec.TensorSpec(
            shape=tensor_shape.TensorShape([None]).concatenate(
              self._field.shape),
            dtype=self._field.dtype,
            name=f'{self._field.name}_values'),
          [
            tensor_spec.TensorSpec(
              shape=tensor_shape.TensorShape([None]),
              dtype=dtypes.int32,
              name=f'{self._field.name}_splits_{i}')
            for i in xrange(self._field.ragged_rank)])

      def _to_components(self, value):
        return value

      def _from_components(self, components):
        return components

      def _batch(self, batch_size):
        r'''Batching of values.
        '''
        del batch_size
        if self._field.ragged_rank == 0:
          raise ValueError(
            f'Field {self._field.name} can not be batched twice')
        raise ValueError(
          f'List field {self._field.name} can not be batched directly')

      def _unbatch(self):
        r'''Unbatching of values.
        '''
        if self._field.ragged_rank == 0:
          return tensor_spec.TensorSpec(
            shape=self._field.shape[1:],
            dtype=self._field.dtype,
            name=self._field.name)
        raise ValueError(
          f'List field {self._field.name} can not be unbatched directly')

      def _to_legacy_output_types(self):
        r'''Output data types for legacy dataset API.
        '''
        if self._field.ragged_rank == 0:
          return self._field.dtype
        return DataFrame.Value(
          self._field.dtype,
          [dtypes.int32 for i in xrange(self._field.ragged_rank)])

      def _to_legacy_output_shapes(self):
        r'''Output shapes for legacy dataset API.
        '''
        if self._field.ragged_rank == 0:
          return tensor_shape.TensorShape([None]).concatenate(
            self._field.shape)
        return DataFrame.Value(
          tensor_shape.TensorShape([None]).concatenate(self._field.shape),
          [
            tensor_shape.TensorShape([None])
            for i in xrange(self._field.ragged_rank)])

      def _to_legacy_output_classes(self):
        r'''Output classes for legacy dataset API.
        '''
        if self._field.ragged_rank == 0:
          return ops.Tensor
        return DataFrame.Value(
          ops.Tensor,
          [ops.Tensor for i in xrange(self._field.ragged_rank)])

    def __init__(
        self, name, dtype=None, ragged_rank=None, shape=None):
      self._name = name
      if dtype is None:
        self._dtype = dtype
      else:
        try:
          self._dtype = dtypes.as_dtype(dtype)
        except TypeError:
          if dtype == np.object_:
            self._dtype = dtypes.as_dtype(np.object)
          else:
            raise
      self._ragged_rank = ragged_rank
      if shape:
        shape = tensor_shape.TensorShape(shape)
        for d in shape:
          if d.value is None:
            raise ValueError(
              f'Field {name} has incomplete shape: {shape}')
        if ragged_rank is not None and ragged_rank > 1:
          raise ValueError(
            f'Field {name} is a nested list ({ragged_rank}) '
            f'with shape {shape}')
      else:
        shape = tensor_shape.TensorShape([])
      self._shape = shape

    @property
    def name(self):
      return self._name

    @property
    def incomplete(self):
      return self.dtype is None or self.ragged_rank is None

    @property
    def dtype(self):
      return self._dtype

    @property
    def ragged_rank(self):
      return self._ragged_rank

    @property
    def shape(self):
      return self._shape

    def __repr__(self):
      if self._dtype is None:
        dtypestr = 'unkown'
      else:
        dtypestr = self._dtype.name
        if self._ragged_rank is None:
          dtypestr = f'unkown<{dtypestr}>'
        else:
          if self._ragged_rank > 1:
            dtypestr = f'list^{self._ragged_rank}<{dtypestr}>'
          elif self._ragged_rank > 0:
            dtypestr = f'list<{dtypestr}>'
      shapestr = str(self._shape)
      return f'{self._name} (dtype={dtypestr}, shape={shapestr})'

    def map(self, func, rank=None):
      if rank is None:
        rank = self.ragged_rank
      if self.incomplete:
        raise ValueError(
          f'Field {self} is incomplete, please specify dtype and ragged_rank')
      if rank == 0:
        return func(0)
      return DataFrame.Value(
        func(0),
        [func(i + 1) for i in xrange(rank)])

    def build_spec(self):
      return DataFrame.Field.Spec(self)

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
      return self.map(
        lambda i: (
          tensor_shape.TensorShape([None]).concatenate(self._shape) if i == 0
          else tensor_shape.TensorShape([None])))

    @property
    def output_specs(self):
      shape = tensor_shape.TensorShape([None]).concatenate(self._shape)
      specs = [tensor_spec.TensorSpec(shape, dtype=self._dtype)]
      specs += [
        tensor_spec.TensorSpec([None], dtype=dtypes.int32)
        for _ in xrange(self._ragged_rank)]
      return specs

  # pylint: disable=inherit-non-class
  class Value(
      collections.namedtuple(
        'DataFrameValue',
        ['values', 'nested_row_splits'])):
    r'''A structure represents a value in DataFrame.
    '''
    def __new__(cls, values, nested_row_splits=None):
      if nested_row_splits is None:
        nested_row_splits = tuple()
      else:
        nested_row_splits = tuple(nested_row_splits)
      return super(DataFrame.Value, cls).__new__(
        cls, values, nested_row_splits)

    def __repr__(self):
      return f'{{{self.values}, splits={self.nested_row_splits}}}'

    def to_list(self):
      result = self.values.tolist()
      for rank in reversed(range(len(self.nested_row_splits))):
        row_splits = self.nested_row_splits[rank]
        result = [
          result[row_splits[i]:row_splits[i + 1]]
          for i in range(len(row_splits) - 1)
        ]
      return result

    def to_tensor(self, name=None):
      if len(self.nested_row_splits) == 0:
        return self.values
      if len(self.nested_row_splits) == 1 and self.values.shape.ndims > 1:
        return self.values
      default_value = array_ops.zeros((), self.values.dtype)
      shape_tensor = constant_op.constant(-1, dtype=dtypes.int32)
      return gen_ragged_conversion_ops.ragged_tensor_to_tensor(
        shape=shape_tensor,
        values=self.values,
        default_value=default_value,
        row_partition_types=['ROW_SPLITS' for _ in self.nested_row_splits],
        row_partition_tensors=self.nested_row_splits,
        name=name)

    def to_sparse(self, name=None):
      if len(self.nested_row_splits) == 0:
        return self.values
      if len(self.nested_row_splits) == 1 and self.values.shape.ndims > 1:
        return self.values
      sparse_value = gen_ragged_conversion_ops.ragged_tensor_to_sparse(
        self.nested_row_splits, self.values, name=name)
      return sparse_tensor.SparseTensor(
        sparse_value.sparse_indices,
        sparse_value.sparse_values,
        sparse_value.sparse_dense_shape)

  @classmethod
  def parse(cls, features, pad=False):
    r'''Convert DataFrame values to tensors or sparse tensors.
    '''
    if isinstance(features, dict):
      return {
        f: cls.parse(
          features[f],
          (pad[f] if f in pad else False) if isinstance(pad, dict) else pad)
        for f in features}
    if isinstance(features, DataFrame.Value):
      if pad:
        return features.to_tensor()
      return features.to_sparse()
    if isinstance(features, ops.Tensor):
      return features
    raise ValueError(f'{features} not supported')

  @classmethod
  def to_sparse(cls, features):
    r'''Convert DataFrame values to tensors or sparse tensors.
    '''
    logging.warning('to_sparse is deprecated, use parse instead')
    return cls.parse(features)

  @classmethod
  def unbatch_and_to_sparse(cls, features):
    r'''Unbatch and convert a row of DataFrame to tensors or sparse tensors.
    '''
    logging.warning('1-batch reading is bad for efficiency')
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
      return features
    raise ValueError(f'{features} not supported for transformation')


def to_sparse(num_parallel_calls=None):
  r'''Convert values to tensors or sparse tensors from input dataset.
  '''
  return parse(num_parallel_calls=num_parallel_calls)


def parse(num_parallel_calls=None, pad=False):
  r'''Convert values to tensors or sparse tensors from input dataset.
  '''
  def _apply_fn(dataset):
    return dataset.map(
      lambda t: DataFrame.parse(t, pad=pad),
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


def input_fields(input_dataset, fields=None):
  r'''Fetch and validate fields from input dataset.
  '''
  if fields is None:
    ds = input_dataset
    while ds:
      if hasattr(ds, 'fields'):
        fields = ds.fields
        break
      if not hasattr(ds, '_input_dataset'):
        break
      ds = ds._input_dataset  # pylint: disable=protected-access
  if not fields:
    raise ValueError('`fields` must be specified')
  if not isinstance(fields, (tuple, list)):
    raise ValueError('`fields` must be a list of `hb.data.DataFrame.Field`.')
  for f in fields:
    if not isinstance(f, DataFrame.Field):
      raise ValueError(f'{f} must be `hb.data.DataFrame.Field`.')
  return fields

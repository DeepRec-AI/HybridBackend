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
from six import string_types as string
from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
from tensorflow.python.data.ops import dataset_ops
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
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging


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
        if self._field.restore_idx_field is not None:
          components.set_deduplicated_idx(self._field.restore_idx_field)
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
        self, name, dtype=None, ragged_rank=None, shape=None,
        default_value=None):
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
      self._default_value = default_value
      self._restore_idx_field = None

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

    @property
    def default_value(self):
      return self._default_value

    @property
    def restore_idx_field(self):
      return self._restore_idx_field

    def set_restore_idx_field(self, field):
      self._restore_idx_field = field

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
      defaultstr = str(self._default_value)
      return (
        f'{self._name} ('
        f'dtype={dtypestr}, '
        f'shape={shapestr}, '
        f'default_value={defaultstr})')

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
      cls._deduplicated_idx = None
      return super(DataFrame.Value, cls).__new__(
        cls, values, nested_row_splits)

    def __repr__(self):
      return f'{{{self.values}, splits={self.nested_row_splits}}}'

    def _create_gather_indices(self, restore_idx):
      r'''construct the indices utilized in gather_nd
      '''
      outer_size_indices = math_ops.range(
        math_ops.cast(
          array_ops.shape(restore_idx)[0],
          dtype=restore_idx.dtype))
      outer_size_indices = array_ops.repeat_with_axis(
        array_ops.reshape(outer_size_indices, [-1, 1]),
        repeats=array_ops.shape(restore_idx)[1:2], axis=-1)
      return array_ops.concat(
        [array_ops.reshape(outer_size_indices, [-1, 1]),
         array_ops.reshape(restore_idx, [-1, 1])], axis=-1)

    @property
    def deduplicated_idx(self):
      return self._deduplicated_idx

    def set_deduplicated_idx(self, deduplicated_idx):
      self._deduplicated_idx = deduplicated_idx

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

    def restore_deduplicated_to_tensor(self, restore_idx, name=None):
      r'''Restore ragged tensors from deduplication
        and convert it to tf.SparseTensor.
      '''
      if (len(self.nested_row_splits) == 0
          or (len(self.nested_row_splits) == 1
              and self.values.shape.ndims > 1)):
        restore_idx = self._create_gather_indices(restore_idx)
        return array_ops.gather_nd(indices=restore_idx, params=self.values)

      restore_idx = self._create_gather_indices(restore_idx)
      ragged_values = ragged_tensor.RaggedTensor.from_nested_row_splits(
        self.values, self.nested_row_splits)
      restored_ragged_tensor = ragged_gather_ops.gather_nd(
        indices=restore_idx, params=ragged_values)
      default_value = array_ops.zeros((), restored_ragged_tensor.dtype)
      return restored_ragged_tensor.to_tensor(
        default_value=default_value, name=name)

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

    def restore_deduplicated_to_sparse(self, restore_idx, name=None):
      r'''Restore ragged tensors from deduplication
        and convert it to tf.sparse.SparseTensor.
      '''
      if (len(self.nested_row_splits) == 0
          or (len(self.nested_row_splits) == 1
              and self.values.shape.ndims > 1)):
        restore_idx = self._create_gather_indices(restore_idx)
        return array_ops.gather_nd(indices=restore_idx, params=self.values)

      restore_idx = self._create_gather_indices(restore_idx)
      ragged_values = ragged_tensor.RaggedTensor.from_nested_row_splits(
        self.values, self.nested_row_splits)
      restored_ragged_tensor = ragged_gather_ops.gather_nd(
        indices=restore_idx, params=ragged_values)
      if isinstance(restored_ragged_tensor, ragged_tensor.RaggedTensor):
        return restored_ragged_tensor.to_sparse(name=name)
      return restored_ragged_tensor

  @classmethod
  def parse(cls, features, pad=False, restore_idx=None):
    r'''Convert DataFrame values to tensors or sparse tensors.
    '''
    if isinstance(features, dict):
      map_feat_to_restore_idx = {}
      parse_res = {}
      for fname, fval in features.items():
        if (isinstance(fval, DataFrame.Value)
            and fval.deduplicated_idx is not None):
          map_feat_to_restore_idx[fname] = fval.deduplicated_idx.name

      for fname in set(map_feat_to_restore_idx.values()):
        parse_res[fname] = cls.parse(
          features[fname], True)

      for fname in features:
        if fname in map_feat_to_restore_idx:
          parse_res[fname] = cls.parse(
            features[fname],
            (pad[fname] if fname in pad else False)
            if isinstance(pad, dict) else pad,
            restore_idx=parse_res[map_feat_to_restore_idx[fname]])
        elif fname not in parse_res:
          parse_res[fname] = cls.parse(
            features[fname],
            (pad[fname] if fname in pad else False)
            if isinstance(pad, dict) else pad)
      restore_idx_names = set(map_feat_to_restore_idx.values())
      for fname in restore_idx_names:
        del parse_res[fname]
      return parse_res
    if (isinstance(features, DataFrame.Value)
        and features.deduplicated_idx is not None):
      if restore_idx is None:
        raise ValueError(
          f'{features} requires a restore_idx to recover deduplication')
      if not isinstance(restore_idx, ops.Tensor):
        raise TypeError(
          f'{restore_idx} shall be a tf.Tensor')
      if pad:
        if isinstance(pad, bool):
          return features.restore_deduplicated_to_tensor(restore_idx)
        output = features.restore_deduplicated_to_tensor(restore_idx)
        output_shape = array_ops.shape(output)
        paddings = [[0, d - output_shape[i]] for i, d in enumerate(pad)]
        return array_ops.pad(output, paddings, 'CONSTANT', constant_values=0)
      return features.restore_deduplicated_to_sparse(restore_idx)
    if isinstance(features, DataFrame.Value):
      if pad:
        if isinstance(pad, bool):
          return features.to_tensor()
        output = features.to_tensor()
        output_shape = array_ops.shape(output)
        paddings = [[0, d - output_shape[i]] for i, d in enumerate(pad)]
        return array_ops.pad(output, paddings, 'CONSTANT', constant_values=0)
      return features.to_sparse()
    if isinstance(features, ops.Tensor):
      return features
    raise ValueError(f'{features} not supported')

  @classmethod
  def populate_defaults(cls, features, all_fields, batch_size):
    r'''Populate default values.
    '''
    if not isinstance(features, dict):
      raise ValueError('Inputs should be a dict')
    populated = dict(features)
    for f in all_fields:
      if f.name not in features and f.default_value is not None:
        if batch_size is None:
          populated[f.name] = array_ops.expand_dims(
            ops.convert_to_tensor(f.default_value, dtype=f.dtype),
            axis=0)
        else:
          if isinstance(f.default_value, sparse_tensor.SparseTensor):
            indices_dtype = f.default_value.indices.dtype
            indices_car = math_ops.cast(
              array_ops.reshape(
                array_ops.tile(
                  math_ops.range(batch_size),
                  [f.default_value.indices.shape[0]]),
                [-1, 1]),
              indices_dtype)
            indices_cdr = array_ops.tile(
              f.default_value.indices, [batch_size, 1])
            batched_indices = array_ops.concat(
              [indices_car, indices_cdr], axis=1)
            batched_values = array_ops.tile(
              f.default_value.values, [batch_size])
            batched_dense_shape = array_ops.concat(
              [ops.convert_to_tensor([batch_size], indices_dtype),
               f.default_value.dense_shape],
              axis=0)
            populated[f.name] = sparse_tensor.SparseTensor(
              indices=batched_indices,
              values=batched_values,
              dense_shape=batched_dense_shape)
          else:
            value = ops.convert_to_tensor(f.default_value, dtype=f.dtype)
            populated[f.name] = array_ops.tile(
              array_ops.expand_dims(value, axis=0),
              tensor_shape.TensorShape([batch_size]).concatenate(
                value.shape))
    return populated

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


def populate_defaults(all_fields, batch_size, num_parallel_calls=None):
  r'''Populate default values.
  '''
  def _apply_fn(dataset):
    return dataset.map(
      lambda t: DataFrame.populate_defaults(t, all_fields, batch_size),
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


def build_fields(filename, fn, fields=None, lower=False):
  r'''Get fields from a file.

  Args:
    filename: Path of the file.
    fields: Existing field definitions or field names.
    lower: Convert field name to lower case if not found.

  Returns:
    Field definitions.
  '''
  logging.info(f'Reading fields from {filename} ...')
  all_field_tuples = fn(filename)  # pylint: disable=c-extension-no-member
  all_fields = {
    f[0]: {'dtype': f[1], 'ragged_rank': f[2]}
    for f in all_field_tuples}
  if fields is None:
    fields = all_fields.keys()
  fields = tuple(fields)
  new_fields = []
  for f in fields:
    if isinstance(f, DataFrame.Field):
      if lower and f.name not in all_fields:
        f = DataFrame.Field(
          f.name.lower(),
          dtype=f.dtype,
          shape=f.shape,
          ragged_rank=f.ragged_rank)
      if f.name not in all_fields:
        if f.default_value is None:
          raise ValueError(
            f'Field {f.name} not found in file {filename}')
      dtype = f.dtype

      if f.name in all_fields:
        actual_dtype = np.dtype(all_fields[f.name]['dtype'])
        if dtype is None:
          dtype = actual_dtype
        elif dtype != actual_dtype:
          raise ValueError(
            f'Field {f.name} dtype should be {actual_dtype} not {dtype}')
        ragged_rank = f.ragged_rank
        actual_ragged_rank = all_fields[f.name]['ragged_rank']
        if ragged_rank is None:
          ragged_rank = actual_ragged_rank
        elif ragged_rank != actual_ragged_rank:
          raise ValueError(
            f'Field {f.name} ragged_rank should be {actual_ragged_rank} '
            f'not {ragged_rank}')
      else:
        if f.default_value is None:
          raise ValueError(
            f'Field {f.name} not found in file {filename}')
        if isinstance(f.default_value, sparse_tensor.SparseTensor):
          actual_dtype = np.dtype(f.default_value.dtype)
          if dtype is None:
            dtype = actual_dtype
          elif dtype != actual_dtype:
            raise ValueError(
              f'Field {f.name} dtype should be {actual_dtype} not {dtype}')
        elif isinstance(f.default_value, ops.Tensor):
          actual_dtype = np.dtype(f.default_value.dtype)
          if dtype is None:
            dtype = actual_dtype
          elif dtype != actual_dtype:
            raise ValueError(
              f'Field {f.name} dtype should be {actual_dtype} not {dtype}')
          actual_ragged_rank = 0
          if ragged_rank is None:
            ragged_rank = actual_ragged_rank
          elif ragged_rank != actual_ragged_rank:
            raise ValueError(
              f'Field {f.name} ragged_rank should be {actual_ragged_rank} '
              f'not {ragged_rank}')
        else:
          try:
            with ops.name_scope('default_values/'):
              _ = ops.convert_to_tensor(
                f.default_value, dtype=dtype)
          except (TypeError, ValueError) as ex:
            raise ValueError(
              f'Field {f.name} default_value {f.default_value} '
              f'should be a SparseTensor or Tensor: {ex}') from ex
      f = DataFrame.Field(
        f.name,
        dtype=dtype,
        ragged_rank=ragged_rank,
        shape=f.shape,
        default_value=None if f.name in all_fields else f.default_value)
      new_fields.append(f)
      continue
    if not isinstance(f, string):
      raise ValueError(
        f'Field {f} is not a DataFrame.Field or a string')
    if lower and f not in all_fields:
      f = f.lower()
    if f not in all_fields:
      raise ValueError(
        f'Field {f} is not found in the file {filename}')
    new_fields.append(DataFrame.Field(
      f,
      dtype=np.dtype(all_fields[f]['dtype']),
      ragged_rank=all_fields[f]['ragged_rank'],
      shape=None))
  return tuple(new_fields)


def build_filenames_and_fields(filenames, fn, fields, lower=False):
  r'''Check and fetch filenames and fields.

  Args:
    filenames: List of file names.
    fields: Existing field definitions or field names.
    lower: Convert field name to lower case if not found.

  Returns:
    Validated file names and fields.
  '''
  if isinstance(filenames, string):
    filenames = [filenames]
    fields = build_fields(filenames[0], fn, fields, lower=lower)
  elif isinstance(filenames, (tuple, list)):
    for f in filenames:
      if not isinstance(f, string):
        raise ValueError(f'{f} in `filenames` must be a string')
    fields = build_fields(filenames[0], fn, fields, lower=lower)
  elif isinstance(filenames, dataset_ops.Dataset):
    if filenames.output_types != dtypes.string:
      raise TypeError(
        '`filenames` must be a `tf.data.Dataset` of `tf.string` elements.')
    if not filenames.output_shapes.is_compatible_with(
        tensor_shape.TensorShape([])):
      raise ValueError(
        '`filenames` must be a `tf.data.Dataset` of scalar `tf.string` '
        'elements.')
    if fields is None:
      raise ValueError('`fields` must be specified.')
    if not isinstance(fields, (tuple, list)):
      raise ValueError('`fields` must be a list of `hb.data.DataFrame.Field`.')
    for f in fields:
      if not isinstance(f, DataFrame.Field):
        raise ValueError(f'Field {f} must be `hb.data.DataFrame.Field`.')
      if f.incomplete:
        raise ValueError(
          f'Field {f} is incomplete, please specify dtype and ragged_rank')
  elif isinstance(filenames, ops.Tensor):
    if filenames.dtype != dtypes.string:
      raise TypeError(
        '`filenames` must be a `tf.Tensor` of `tf.string`.')
    if fields is None:
      raise ValueError('`fields` must be specified.')
    if not isinstance(fields, (tuple, list)):
      raise ValueError('`fields` must be a list of `hb.data.DataFrame.Field`.')
    for f in fields:
      if not isinstance(f, DataFrame.Field):
        raise ValueError(f'Field {f} must be `hb.data.DataFrame.Field`.')
      if f.incomplete:
        raise ValueError(
          f'Field {f} is incomplete, please specify dtype and ragged_rank')
  else:
    raise ValueError(
      f'`filenames` {filenames} must be a `tf.data.Dataset` of scalar '
      '`tf.string` elements or can be converted to a `tf.Tensor` of '
      '`tf.string`.')

  if not isinstance(filenames, dataset_ops.Dataset):
    filenames = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    filenames = array_ops.reshape(filenames, [-1], name='filenames')
    filenames = dataset_ops.Dataset.from_tensor_slices(filenames)
  return filenames, fields

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

r'''Feature columns for embedding lookup.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context as _context
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops

from hybridbackend.tensorflow.feature_column.embedding_backend import \
  EmbeddingBackend
from hybridbackend.tensorflow.feature_column.embedding_lookup import \
  EmbeddingLookup
from hybridbackend.tensorflow.framework.ops import GraphKeys


class StateManagerImpl(fc._StateManagerImpl):  # pylint: disable=protected-access
  r'''Manages the state of DenseFeatures.
  '''
  def create_variable(
      self,
      feature_column,
      name,
      shape,
      dtype=None,
      trainable=True,
      use_resource=True,
      initializer=None):
    r'''Creates a new variable.
    '''
    if name in self._cols_to_vars_map[feature_column]:
      raise ValueError('Variable already exists.')

    impl = EmbeddingBackend.get()
    collections = impl.collections(feature_column)
    if impl.sharded(feature_column):
      collections.append(GraphKeys.SHARDED_VARIABLES)
    var = impl.build(
      feature_column,
      name,
      shape,
      dtype=dtype,
      trainable=self._trainable and trainable,
      use_resource=use_resource,
      initializer=initializer,
      collections=collections,
      layer=self._layer)
    self._cols_to_vars_map[feature_column][name] = var
    return var


class EmbeddingColumn(fc.EmbeddingColumn):
  r'''Feature column for dense features.
  '''
  def __new__(
      cls,
      categorical_column,
      dimension,
      combiner,
      initializer,
      ckpt_to_load_from,
      tensor_name_in_ckpt,
      max_norm,
      trainable,
      **kwargs):
    r'''Constructor.
    '''
    self = super(EmbeddingColumn, cls).__new__(
      cls,
      categorical_column,
      dimension,
      combiner,
      initializer,
      ckpt_to_load_from,
      tensor_name_in_ckpt,
      max_norm,
      trainable,
      **kwargs)
    # pylint: disable=protected-access
    self._impl = EmbeddingBackend.get()
    self._lookup_sparse = EmbeddingLookup(self)
    # pylint: enable=protected-access
    return self

  def _get_dense_tensor_internal_helper(
      self, sparse_tensors, embedding_weights):
    r'''Private method that follows the signature of get_dense_tensor.
    '''
    if self.ckpt_to_load_from is not None:
      self._impl.init_from_checkpoint(
        self,
        self.ckpt_to_load_from,
        self.tensor_name_in_ckpt,
        embedding_weights)

    return self._lookup_sparse(
      embedding_weights, sparse_tensors,
      name=f'{self.name}_lookup')

  def _get_dense_tensor_internal(self, sparse_tensors, state_manager):
    r'''Private method that follows the signature of get_dense_tensor.
    '''
    embedding_weights = state_manager.get_variable(
      self, self._impl.weight_name(self))
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def output_shape(self, inputs):
    r'''Tuple of column output shape.
    '''
    inputs_shape = array_ops.shape(inputs)[:-1]
    num_elements = self.variable_shape.num_elements()
    actual_inputs_shape = array_ops.concat(
      [inputs_shape, array_ops.constant(num_elements, shape=[1])],
      0, name='actual_inputs_shape')
    return actual_inputs_shape

  def create_state(self, state_manager):
    r'''Uses the `state_manager` to create state for the FeatureColumn.
    '''
    num_buckets = getattr(self.categorical_column, 'num_buckets',
                          self.categorical_column._num_buckets)  # pylint: disable=protected-access
    embedding_shape = (num_buckets, self.dimension)
    embedding_name = self._impl.weight_name(self)
    with ops.device(self._impl.device(self)):
      return state_manager.create_variable(
        self,
        name=embedding_name,
        shape=embedding_shape,
        dtype=self._impl.dtype(self),
        trainable=self.trainable,
        use_resource=True,
        initializer=self.initializer)


class SharedEmbeddingColumn(fc.SharedEmbeddingColumn):
  r'''Feature column for dense features.
  '''
  def __new__(
      cls,
      categorical_column,
      shared_embedding_column_creator,
      combiner,
      max_norm,
      **kwargs):
    r'''Constructor.
    '''
    self = super(SharedEmbeddingColumn, cls).__new__(
      cls,
      categorical_column,
      shared_embedding_column_creator,
      combiner,
      max_norm,
      **kwargs)
    # pylint: disable=protected-access
    self._impl = EmbeddingBackend.get()
    self._lookup_sparse = EmbeddingLookup(self)
    # pylint: enable=protected-access
    return self

  def _get_dense_tensor_internal(self, transformation_cache, state_manager):
    r'''Private method that follows the signature of _get_dense_tensor.
    '''
    with ops.name_scope(None, default_name=self.name):
      # Get sparse IDs and weights.
      sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
      # Return embedding lookup result.
      return self._lookup_sparse(
        self.shared_embedding_column_creator.embedding_weights,
        sparse_tensors,
        name=f'{self.name}_lookup')


class SharedEmbeddingColumnCreator(fc.SharedEmbeddingColumnCreator):
  r'''Creator to create shared embedding columns.
  '''
  def __init__(self,
               dimension,
               initializer,
               ckpt_to_load_from,
               tensor_name_in_ckpt,
               num_buckets,
               trainable,
               name='shared_embedding_column_creator',
               **kwargs):
    super().__init__(
      dimension, initializer, ckpt_to_load_from, tensor_name_in_ckpt,
      num_buckets, trainable, name=name, **kwargs)
    self._impl = EmbeddingBackend.get()
    self._fc = None

  def __call__(self, categorical_column, combiner, max_norm, **kwargs):
    self._fc = SharedEmbeddingColumn(
      categorical_column, self, combiner, max_norm, **kwargs)
    return self._fc

  @property
  def embedding_weights(self):
    r'''Get or create embedding weights.
    '''
    key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    if key not in self._embedding_weights:
      embedding_shape = (self._num_buckets, self._dimension)
      collections = self._impl.collections(self._fc)
      if self._impl.sharded(self._fc):
        collections.append(GraphKeys.SHARDED_VARIABLES)
      with ops.device(self._impl.device(self._fc)):
        var = self._impl.build(
          self._fc,
          self._impl.weight_name(self._fc),
          embedding_shape,
          dtype=self._impl.dtype(self._fc),
          trainable=self._trainable,
          use_resource=True,
          initializer=self._initializer,
          collections=collections)
      if self._ckpt_to_load_from is not None:
        self._impl.init_from_checkpoint(
          self._fc,
          self._ckpt_to_load_from,
          self._tensor_name_in_ckpt,
          var)
      self._embedding_weights[key] = var
    return self._embedding_weights[key]


def shared_embedding_columns(categorical_columns,
                             dimension,
                             combiner='mean',
                             initializer=None,
                             shared_embedding_collection_name=None,
                             ckpt_to_load_from=None,
                             tensor_name_in_ckpt=None,
                             max_norm=None,
                             trainable=True,
                             **kwargs):
  r'''List of dense columns that convert from sparse, categorical input.

  This is similar to `embedding_column`, except that it produces a list of
  embedding columns that share the same embedding weights.

  Args:
    categorical_columns: `_CategoricalColumn`s created by a
      `categorical_column_with_*` function. This column produces
      the sparse IDs that are inputs to the embedding lookup.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    ckpt_to_load_from: String representing checkpoint name/pattern
      from which to restore column weights. Required
      if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from
      which to restore the column weights. Required if `ckpt_to_load_from` is
      not `None`.
    max_norm: If not `None`, embedding values are l2-normalized to this value.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    A list of embedding columns that share the same embedding weights.
  '''
  if _context.executing_eagerly():
    raise RuntimeError('shared_embedding_columns are not supported when eager '
                       'execution is enabled.')

  if (dimension is None) or (dimension < 1):
    raise ValueError(f'Invalid dimension {dimension}.')
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
      mean=0.0, stddev=1. / math.sqrt(dimension))

  # Sort the columns so the default collection name is deterministic even if the
  # user passes columns from an unsorted collection, such as dict.values().
  sorted_columns = sorted(categorical_columns, key=lambda x: x.name)

  c0 = sorted_columns[0]
  num_buckets = c0.num_buckets
  if not isinstance(c0, fc.CategoricalColumn):
    raise ValueError(
      'All categorical_columns must be subclasses of CategoricalColumn. '
      f'Given: {c0}, of type: {type(c0)}')
  if isinstance(c0, fc.WeightedCategoricalColumn):
    c0 = c0.categorical_column
  for c in sorted_columns[1:]:
    if isinstance(c, fc.WeightedCategoricalColumn):
      c = c.categorical_column
    if not isinstance(c, type(c0)):
      raise ValueError(
        'To use shared_embedding_column, all categorical_columns must have '
        'the same type, or be weighted_categorical_column of the same type. '
        f'Given column: {c0} of type: {type(c0)} does not match given column:'
        f' {c} of type: {type(c)}')
    if num_buckets != c.num_buckets:
      raise ValueError(
        'To use shared_embedding_column, all categorical_columns must have '
        f'the same number of buckets. Given column: {c0} with buckets: '
        f'{num_buckets} does not match column: {c} with buckets: '
        f'{c.num_buckets}')

  if not shared_embedding_collection_name:
    shared_embedding_collection_name = '_'.join(c.name for c in sorted_columns)
    shared_embedding_collection_name += '_shared_embedding'

  column_creator = SharedEmbeddingColumnCreator(
    dimension, initializer, ckpt_to_load_from, tensor_name_in_ckpt,
    num_buckets, trainable, shared_embedding_collection_name,
    **kwargs)

  result = []
  for column in categorical_columns:
    result.append(
      column_creator(
        categorical_column=column,
        combiner=combiner,
        max_norm=max_norm))

  return result


def embedding_column(
    categorical_column,
    dimension,
    combiner='mean',
    initializer=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    max_norm=None,
    trainable=True,
    **kwargs):
  r'''`DenseColumn` that converts from sparse, categorical input.

  Args:
    categorical_column: A `_CategoricalColumn` created by a
      `categorical_column_with_*` function. This column produces
      the sparse IDs that are inputs to the embedding lookup.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    ckpt_to_load_from: String representing checkpoint name/pattern
      from which to restore column weights. Required
      if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from
      which to restore the column weights. Required if `ckpt_to_load_from` is
      not `None`.
    max_norm: If not `None`, embedding values are l2-normalized to this value.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    An embedding column.
  '''
  if (dimension is None) or (dimension < 1):
    raise ValueError(f'Invalid dimension {dimension}.')
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. '
                     f'Embedding of column_name: {categorical_column.name}')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
      mean=0.0, stddev=1 / math.sqrt(dimension))

  shared_name = kwargs.pop('shared_name', None)
  if shared_name is None:
    return EmbeddingColumn(
      categorical_column,
      dimension,
      combiner,
      initializer,
      ckpt_to_load_from,
      tensor_name_in_ckpt,
      max_norm,
      trainable,
      **kwargs)

  column_creator = SharedEmbeddingColumnCreator(
    dimension, initializer, ckpt_to_load_from, tensor_name_in_ckpt,
    categorical_column.num_buckets, trainable, shared_name,
    **kwargs)
  return column_creator(
    categorical_column=categorical_column,
    combiner=combiner,
    max_norm=max_norm)

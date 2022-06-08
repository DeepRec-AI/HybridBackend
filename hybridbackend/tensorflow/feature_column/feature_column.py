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

import sys
import threading

from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops

from hybridbackend.tensorflow.feature_column.embedding_backend import \
  EmbeddingBackend
from hybridbackend.tensorflow.feature_column.embedding_lookup import \
  EmbeddingLookup
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys


class PatchEmbeddingColumn(object):  # pylint: disable=useless-object-inheritance
  r'''Context manager that patches fc.EmbeddingColumn.
  '''
  _lock = threading.Lock()
  _stack_depth = 0

  def __enter__(self):
    with PatchEmbeddingColumn._lock:
      PatchEmbeddingColumn._stack_depth += 1
      if PatchEmbeddingColumn._stack_depth <= 1:

        def decorate_get_raw_feature(fn):  # pylint: disable=unused-argument
          r'''w/o expanding dims when rank equals one.
          '''
          def decorated(col, key):
            raw_feature = col._features[key]  # pylint: disable=protected-access
            feature_tensor = \
              sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(raw_feature)
            rank = feature_tensor.get_shape().ndims
            if rank is not None:
              if rank == 0:
                raise ValueError(
                  f'Feature (key: {key}) cannot have rank 0.'
                  f'Given: {feature_tensor}')
              return feature_tensor
            # Handle dynamic rank.
            with ops.control_dependencies([
                check_ops.assert_positive(
                  array_ops.rank(feature_tensor),
                  message=f'Feature (key: {key}) cannot have rank 0.'
                          f'Given: {feature_tensor}')]):
              return feature_tensor
          return decorated
        self._prev_get_raw_feature = \
          fc.FeatureTransformationCache._get_raw_feature_as_tensor
        fc.FeatureTransformationCache._get_raw_feature_as_tensor = \
          decorate_get_raw_feature(self._prev_get_raw_feature)

      return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    with PatchEmbeddingColumn._lock:
      if PatchEmbeddingColumn._stack_depth <= 1:
        fc.FeatureTransformationCache._get_raw_feature_as_tensor = \
          self._prev_get_raw_feature
      PatchEmbeddingColumn._stack_depth -= 1


def _get_sparse_tensors(
    categorical_column, transformation_cache, state_manager):
  r'''Returns tensor before doing the embedding lookup.
  '''
  if isinstance(categorical_column, fc.SequenceCategoricalColumn):
    categorical_column_type = type(categorical_column)
    raise ValueError(
      f'In embedding_column: {self.name}. '
      'categorical_column must not be of type SequenceCategoricalColumn. '
      'Suggested fix A: If you wish to use DenseFeatures, use a '
      'non-sequence categorical_column_with_*. '
      'Suggested fix B: If you wish to create sequence input, use '
      'SequenceFeatures instead of DenseFeatures. '
      f'Given (type {categorical_column_type}): '
      f'{categorical_column}')

  if isinstance(categorical_column, fc.IdentityCategoricalColumn):
    dynamic_bucketing = (
      Context.get().options.emb_num_buckets_max[
        categorical_column.name])
    if categorical_column.number_buckets >= dynamic_bucketing:
      with PatchEmbeddingColumn():
        return fc.CategoricalColumn.IdWeightPair(
          transformation_cache.get(categorical_column.key, state_manager), None)

  return categorical_column.get_sparse_tensors(
    transformation_cache, state_manager)


def raw_categorical_column(key):
  r'''A `CategoricalColumn` that returns unchecked values.

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
  '''
  return fc.categorical_column_with_identity(key, num_buckets=sys.maxsize)


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
    collections = [ops.GraphKeys.GLOBAL_VARIABLES]
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
  @classmethod
  def build(cls, prototype):
    r'''Build EmbeddingColumn from prototype.
    '''
    if isinstance(prototype, cls):
      return prototype
    if isinstance(prototype, fc.EmbeddingColumn):
      prototype_kwargs = dict(prototype._asdict())  # pylint: disable=protected-access
      return cls(**prototype_kwargs)
    raise ValueError('EmbeddingColumn required')

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
    embedding_weights = self.get_state(state_manager)
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def get_dense_tensor(self, transformation_cache, state_manager):
    r'''Returns tensor after doing the embedding lookup.
    '''
    sparse_tensors = _get_sparse_tensors(
      self.categorical_column, transformation_cache, state_manager)
    return self._get_dense_tensor_internal(sparse_tensors, state_manager)

  def output_shape(self, inputs):
    r'''Tuple of column output shape.
    '''
    inputs_shape = array_ops.shape(inputs)[:-1]
    num_elements = self.variable_shape.num_elements()
    actual_inputs_shape = array_ops.concat(
      [inputs_shape, array_ops.constant(num_elements, shape=[1])],
      0, name='actual_inputs_shape')
    return actual_inputs_shape

  def get_state(self, state_manager):
    r'''Get or create embedding weights.
    '''
    return state_manager.get_variable(self, self._impl.weight_name(self))

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


class SharedEmbeddingColumn(fc_old._SharedEmbeddingColumn, fc.DenseColumn):  # pylint: disable=protected-access
  r'''Feature column for dense features.
  '''
  @classmethod
  def build(cls, prototype):
    r'''Build SharedEmbeddingColumn from prototype.
    '''
    if isinstance(prototype, cls):
      return prototype
    if isinstance(prototype, fc_old._SharedEmbeddingColumn):  # pylint: disable=protected-access
      prototype_kwargs = dict(prototype._asdict())  # pylint: disable=protected-access
      return cls(**prototype_kwargs)
    raise ValueError('SharedEmbeddingColumn required')

  def __new__(
      cls,
      categorical_column,
      dimension,
      combiner,
      initializer,
      shared_embedding_collection_name,
      ckpt_to_load_from,
      tensor_name_in_ckpt,
      max_norm,
      trainable,
      **kwargs):
    r'''Constructor.
    '''
    self = super(SharedEmbeddingColumn, cls).__new__(
      cls,
      categorical_column,
      dimension,
      combiner,
      initializer,
      shared_embedding_collection_name,
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
    embedding_weights = self.get_state(state_manager)
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def get_dense_tensor(self, transformation_cache, state_manager):
    r'''Returns tensor after doing the embedding lookup.
    '''
    sparse_tensors = _get_sparse_tensors(
      self.categorical_column, transformation_cache, state_manager)
    return self._get_dense_tensor_internal(sparse_tensors, state_manager)

  @property
  def variable_shape(self):
    r'''See `DenseColumn` base class.
    '''
    return tensor_shape.TensorShape([self.dimension])

  def output_shape(self, inputs):
    r'''Tuple of column output shape.
    '''
    inputs_shape = array_ops.shape(inputs)[:-1]
    num_elements = self.variable_shape.num_elements()
    actual_inputs_shape = array_ops.concat(
      [inputs_shape, array_ops.constant(num_elements, shape=[1])],
      0, name='actual_inputs_shape')
    return actual_inputs_shape

  def get_state(self, state_manager):
    r'''Get or create embedding weights.
    '''
    del state_manager
    shared_embedding_collection = ops.get_collection(
      self.shared_embedding_collection_name)
    if not shared_embedding_collection:
      raise ValueError('create_state must be called before')
    if len(shared_embedding_collection) > 1:
      raise ValueError(
        f'Collection {shared_embedding_collection} can only contain one '
        'variable. Suggested fix A: Choose a unique name for this '
        'collection. Suggested fix B: Do not add any variables to this '
        'collection. The feature_column library already adds a variable '
        'under the hood.')
    return shared_embedding_collection[0]

  def create_state(self, state_manager):
    r'''Uses the `state_manager` to create state for the FeatureColumn.
    '''
    shared_embedding_collection = ops.get_collection(
      self.shared_embedding_collection_name)
    if shared_embedding_collection:
      if len(shared_embedding_collection) > 1:
        raise ValueError(
          f'Collection {shared_embedding_collection} can only contain one '
          'variable. Suggested fix A: Choose a unique name for this '
          'collection. Suggested fix B: Do not add any variables to this '
          'collection. The feature_column library already adds a variable '
          'under the hood.')
      return shared_embedding_collection[0]

    num_buckets = getattr(self.categorical_column, 'num_buckets',
                          self.categorical_column._num_buckets)  # pylint: disable=protected-access
    embedding_shape = (num_buckets, self.dimension)
    embedding_name = self._impl.weight_name(self)
    with ops.device(self._impl.device(self)):
      embedding_weights = state_manager.create_variable(
        self,
        name=embedding_name,
        shape=embedding_shape,
        dtype=self._impl.dtype(self),
        trainable=self.trainable,
        use_resource=True,
        initializer=self.initializer)
      ops.add_to_collection(self.shared_embedding_collection_name,
                            embedding_weights)
      return embedding_weights

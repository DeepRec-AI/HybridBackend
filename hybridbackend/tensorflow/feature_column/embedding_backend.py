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

r'''Backend of embedding tables.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from hybridbackend.tensorflow.framework.context import Context


class EmbeddingBackend(object):  # pylint: disable=useless-object-inheritance
  r'''Backend for embedding columns.

  An embedding backend manages underlying storage for embedding columns. Data
  scientists can extend this class for customized implementation of embedding
  weights.
  '''
  _registry = {}

  @classmethod
  def register(cls, impl):
    r'''Register implementation.

    Args:
      impl: Implementation to register.
    '''
    cls._registry[impl.NAME] = impl

  @classmethod
  def get(cls):
    r'''Get an instance of registered implementation.

    Returns:
      An instance of registered implementation.
    '''
    backend = Context.get().param(
      'emb_backend', 'DEFAULT',
      env='HB_EMB_BACKEND')
    if backend not in cls._registry:
      raise ValueError(f'emb_backend is invalid: {backend}')
    return cls._registry[backend]

  @abc.abstractmethod
  def build(
      self,
      column,
      name,
      shape,
      dtype=None,
      trainable=True,
      use_resource=True,
      initializer=None,
      collections=None,
      layer=None):
    r'''Creates the embedding lookup weight.

    Args:
      column: An `EmbeddingColumn` for building weight.
      name: Name of the embedding weights.
      shape: Shape of the embedding weights.
      dtype: (Optional.) Data type of the embedding weights.
      trainable: (Optional.) If True, the embedding weights can be trained.
      use_resource: (Optional.) If True, the embedding weights uses resource.
      initializer: (Optional.) Initializer of the embedding weights.
      collections: (Optional.) Collections of the embedding weights.
      layer: `DenseFeatures` layer which manages weights.

    Returns:
      The embedding weight.
    '''

  @abc.abstractmethod
  def init_from_checkpoint(
      self, column, ckpt_dir_or_file, tensor_name_in_ckpt, to_restore):
    r'''Replaces initializers of embedding weights to load from checkpoints.

    Args:
      column: An `EmbeddingColumn`.
      ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint
        or checkpoint file path.
      tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_dir_or_file` from
        which  to restore the column weights. Required if `ckpt_to_load_from`
        is not `None`.
      to_restore: `Tensor` to restore.
    '''

  @abc.abstractmethod
  def lookup(self, column, weight, inputs, sharded=False, buffered=False):
    r'''Lookup for embedding vectors.

    Args:
      column: An `EmbeddingColumn`.
      weight: Embedding weight.
      inputs: Inputs for embedding lookup.
      sharded: If True inputs are sharded.
      buffered: If True initialization would be delayed.

    Returns:
      Embedding vectors from the weight.
    '''

  @abc.abstractmethod
  def update(self, column, weight, indexed_updates):
    r'''Update embedding weight.

    Args:
      column: An `EmbeddingColumn`.
      weight: Embedding weight.
      indexed_updates: An `IndexedSlices` to update weight in specific indices.
    '''

  def buffer_size(self):
    r'''Buffer size of embedding variables for training.
    '''
    return Context.get().param(
      'emb_buffer_size', 0,
      env='HB_EMB_BUFFER_SIZE',
      parser=int)

  def buffer_load_factor(self):
    r'''Buffer load factor of embedding variables for training.
    '''
    return Context.get().param(
      'emb_buffer_load_factor', 0.5,
      env='HB_EMB_BUFFER_LOAD_FACTOR',
      parser=float)

  def num_groups(self):
    r'''Number of embedding column groups for training.
    '''
    return Context.get().param(
      'emb_num_groups', 1,
      env='HB_EMB_NUM_GROUPS',
      parser=int)

  def enable_concat(self):
    r'''If True, concat embedding vectors for training.
    '''
    return Context.get().param(
      'emb_enable_concat', True,
      env='HB_EMB_ENABLE_CONCAT',
      parser=Context.parse_bool)

  def num_buckets(self, column):
    r'''Number of buckets for the column.
    '''
    num_buckets = getattr(
      column.categorical_column, 'num_buckets',
      column.categorical_column._num_buckets)  # pylint: disable=protected-access
    return num_buckets

  def dimension(self, column):
    r'''Dimension of the column.
    '''
    if hasattr(column, 'shared_embedding_column_creator'):
      return column.shared_embedding_column_creator.dimension
    return column.dimension

  @property
  def enable_sharding(self):
    r'''Whether the sharding is enabled.
    '''
    return Context.get().param(
      'enable_sharding', True,
      parser=Context.parse_bool)

  def sharded(self, column):
    r'''Whether the column should be sharded.
    '''
    if Context.get().world_size <= 1:
      return False
    if not self.enable_sharding:
      return False
    batch_size = Context.get().batch_size
    if batch_size < 0 or self.num_buckets(column) is None:
      return True
    if batch_size < self.num_buckets(column):
      return True
    return False

  def unique(self, column):
    r'''Whether inputs for the column is already unique.
    '''
    return Context.get().param(
      'emb_unique', False,
      env='HB_EMB_UNIQUE',
      parser=Context.parse_bool,
      select=column.categorical_column.name)

  def shard_unique(self, column):
    r'''Whether inputs for weight shard of the column is already unique.
    '''
    return Context.get().param(
      'emb_shard_unique', False,
      env='HB_EMB_SHARD_UNIQUE',
      parser=Context.parse_bool,
      select=column.categorical_column.name)

  def device(self, column):
    r'''Device of the column weights.
    '''
    return Context.get().param(
      'emb_device', '',
      env='HB_EMB_DEVICE',
      select=column.categorical_column.name)

  def input_device(self, column):
    r'''Device of embedding lookup inputs.
    '''
    return Context.get().param(
      'emb_input_device', '',
      env='HB_EMB_INPUT_DEVICE',
      select=column.categorical_column.name)

  def dtype(self, column):
    r'''Data type of the column weights.
    '''
    return Context.get().param(
      'emb_dtype', dtypes.float32,
      env='HB_EMB_DTYPE',
      parser=dtypes.as_dtype,
      select=column.categorical_column.name)

  def wire_dtype(self, column):
    r'''Data type of the column for communicaiton.
    '''
    return Context.get().param(
      'emb_wire_dtype', dtypes.float32,
      env='HB_EMB_WIRE_DTYPE',
      parser=dtypes.as_dtype,
      select=column.categorical_column.name)

  def input_dtype(self, column):
    r'''Data type of the column inputs.
    '''
    return Context.get().param(
      'emb_input_dtype', dtypes.int64,
      select=column.categorical_column.name)

  def collections(self, column):
    r'''Collections of the column weights.
    '''
    return Context.get().param(
      'emb_collections', [ops.GraphKeys.GLOBAL_VARIABLES],
      select=column.categorical_column.name)

  def segment_rank(self, column):
    r'''Segment rank of the column weights.
    '''
    return Context.get().param(
      'emb_segment_rank', 0,
      env='HB_EMB_SEGMENT_RANK',
      parser=int,
      select=column.categorical_column.name)

  def weight_name(self, column):
    r'''Name of the column weights.
    '''
    name = f'{column.categorical_column.name}/embedding_weights'
    if self.sharded(column):
      shard = Context.get().rank
      name = f'{name}/part_{shard}'
    return name

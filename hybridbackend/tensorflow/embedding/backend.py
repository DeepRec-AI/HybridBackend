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

from tensorflow.python.saved_model.model_utils import mode_keys

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
    backend = Context.get().options.emb_backend
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
    return Context.get().options.emb_buffer_size

  def buffer_load_factor(self):
    r'''Buffer load factor of embedding variables for training.
    '''
    return Context.get().options.emb_buffer_load_factor

  def num_groups(self):
    r'''Number of embedding column groups for training.
    '''
    return Context.get().options.emb_num_groups

  def enable_concat(self):
    r'''If True, concat embedding vectors for training.
    '''
    return Context.get().options.emb_enable_concat

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
    return column.dimension

  @property
  def enable_sharding(self):
    r'''Whether the sharding is enabled.
    '''
    return not mode_keys.is_predict(Context.get().options.mode)

  def sharded(self, column):
    r'''Whether the column should be sharded.
    '''
    if Context.get().world_size <= 1:
      return False
    if not self.enable_sharding:
      return False
    batch_size = Context.get().options.batch_size
    if batch_size < 0 or self.num_buckets(column) is None:
      return True
    if batch_size < self.num_buckets(column):
      return True
    return False

  def unique(self, column):
    r'''Whether inputs for the column is already unique.
    '''
    return Context.get().options.emb_unique[(
      column.name, column.categorical_column.name)]

  def pad(self, column):
    r'''Whether lookup results of the column should be padded.
    '''
    return Context.get().options.emb_pad[(
      column.name, column.categorical_column.name)]

  def device(self, column):
    r'''Device of the column weights.
    '''
    return Context.get().options.emb_device[(
      column.name, column.categorical_column.name)]

  def input_device(self, column):
    r'''Device of embedding lookup inputs.
    '''
    options = Context.get().options
    return options.emb_input_device[(
      column.name, column.categorical_column.name)]

  def dtype(self, column):
    r'''Data type of the column weights.
    '''
    return Context.get().options.emb_dtype[(
      column.name, column.categorical_column.name)]

  def wire_dtype(self, column):
    r'''Data type of the column for communicaiton.
    '''
    return Context.get().options.emb_wire_dtype[(
      column.name, column.categorical_column.name)]

  def input_dtype(self, column):
    r'''Data type of the column inputs.
    '''
    return Context.get().options.emb_input_dtype[(
      column.name, column.categorical_column.name)]

  def segment_rank(self, column):
    r'''Segment rank of the column weights.
    '''
    options = Context.get().options
    return options.emb_segment_rank[(
      column.name, column.categorical_column.name)]

  def weight_name(self, column):
    r'''Name of the column weights.
    '''
    name = 'embedding_weights'
    if self.sharded(column):
      shard = Context.get().rank
      name = f'{name}/part_{shard}'
    return name

  def weight_shared_name(self, column, var):
    r'''Get shared name of the column weights from an variable.
    '''
    var_name = var.name.split(':')[0]
    if self.sharded(column):
      return var_name.split('/part')[0]
    return var_name

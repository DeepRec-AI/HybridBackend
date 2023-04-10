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

r'''Sharded embedding lookup related classes and functions.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.contrib.layers.python.layers import \
  feature_column as _contrib_feature_column
from tensorflow.python.feature_column.feature_column import \
  _SharedEmbeddingColumn as _shared_emb
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops

from hybridbackend.tensorflow.distribute.collective import Collective
from hybridbackend.tensorflow.distribute.ops import Topology
from hybridbackend.tensorflow.distribute.partition.ops import \
  partition_by_dual_modulo_stage_one
from hybridbackend.tensorflow.distribute.partition.ops import \
  partition_by_dual_modulo_stage_two
from hybridbackend.tensorflow.distribute.partition.ops import \
  partition_by_modulo
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys
from hybridbackend.tensorflow.framework.ops import ModeKeys
from hybridbackend.tensorflow.framework.rewriting import GraphRewriting


class ShardedEmbeddingWeightsRewriting(GraphRewriting):  # pylint: disable=useless-object-inheritance
  r'''Rewriting embedding weights.
  '''
  @classmethod
  def register(cls, rewriting):
    r'''Register implementation.

    Args:
      rewriting: Implementation class to register.
    '''
    return GraphRewriting.register(rewriting)

  @property
  def isdynamic(self):
    r'''Whether embedding weights is dynamic.
    '''
    return False

  @property
  def should_shard(self):
    r'''Whether embedding weights sharding should be enabled.
    '''
    ctx = Context.get()
    return (
      ctx.world_size > 1
      and ctx.options.sharding
      and ctx.options.mode != ModeKeys.PREDICT)

  def _nosharding(self, fn):
    r'''Wraps fn without sharding.
    '''
    def wrapped_fn(*args, **kwargs):
      with Context.scope(sharding=False):
        return fn(*args, **kwargs)
    return wrapped_fn

  def _sharded_embedding_weights_collections(self, collections):
    r'''Build collections for sharded weights.
    '''
    minimal_collections = [
      ops.GraphKeys.GLOBAL_VARIABLES,
      GraphKeys.SHARDED_VARIABLES]
    if collections is None:
      return list(minimal_collections)
    collections_copy = list(collections)
    collections_copy.extend(minimal_collections)
    if self.isdynamic:
      collections_copy.append(GraphKeys.DYNAMIC_VARIABLES)
    return list(set(collections_copy))

  def wraps_build_embedding_weights(self, fn):
    r'''Rewrites weights building.
    '''
    def _wrapped_build_weights(*args, **kwargs):
      r'''Build embedding weights.
      '''
      if not self.should_shard:
        return self.build_unsharded_weights(fn, *args, **kwargs)

      shard_collections = kwargs.get('collections', None)
      shard_collections = self._sharded_embedding_weights_collections(
        shard_collections)
      return self.build_sharded_weights(
        Context.get().rank, Context.get().world_size,
        shard_collections, self._nosharding(fn),
        *args, **kwargs)
    return _wrapped_build_weights

  def build_unsharded_weights(self, fn, *args, **kwargs):
    r'''Build unsharded embedding weights.
    '''
    return fn(*args, **kwargs)

  def build_sharded_weights(
      self, shard, num_shards, shard_collections, fn, *args, **kwargs):
    r'''Build sharded embedding weights.
    '''
    del shard
    del num_shards
    del shard_collections
    return fn(*args, **kwargs)

  @abc.abstractmethod
  def begin(self):
    r'''Rewrites API.
    '''

  @abc.abstractmethod
  def end(self):
    r'''Revert API rewriting.
    '''


class ShardedEmbeddingLookupRewriting(GraphRewriting):
  r'''Embedding lookup rewriting for tf.nn.embedding_lookup.
  '''
  def __init__(self):
    super().__init__()
    self._prev_lookup = None
    self._prev_shared_emb_get_dense_tensor_internal = None
    self._local_world_size = Context.get().local_world_size
    self._num_nodes = Context.get().world_size // self._local_world_size

  def issharded(self, weights):
    r'''Check whether the embedding weights are sharded.
    '''
    sharded_variables = ops.get_default_graph().get_collection_ref(
      GraphKeys.SHARDED_VARIABLES)
    if weights in sharded_variables:
      return True
    if isinstance(weights, (list, tuple)) and len(weights) == 1:
      weights = weights[0]
      if isinstance(weights, ops.Tensor):
        vname = weights.name.split('/read')[0]
        for v in sharded_variables:
          if vname == v.name.split(':')[0]:
            return True
    return False

  def wraps_embedding_lookup(self, fn):
    r'''Rewrites embedding_lookup.
    '''
    def _wrapped_embedding_lookup(params, ids, **kwargs):
      r'''Looks up `ids` in a list of embedding tensors.
      '''
      if not self.issharded(params):
        with Context.scope(sharding=False):
          return fn(params, ids, **kwargs)

      current_device = control_flow_ops.no_op().device
      num_shards = Context.get().world_size
      with Context.scope(sharding=False):
        with ops.device(Context.get().devices[Context.get().rank]):
          ids_shards, ids_sizes, shard_index = partition_by_modulo(
            ids, num_shards, name='shard_partition')
          shard_ids, shard_sizes = Collective.get().alltoall(
            ids_shards, sizes=ids_sizes)
          shard_ids, shard_unique_index = array_ops.unique(
            shard_ids, name='shard_unique')
          if params not in ops.get_collection_ref(GraphKeys.DYNAMIC_VARIABLES):
            shard_ids = shard_ids // num_shards
          with ops.device(current_device):
            embeddings = fn(params, shard_ids, **kwargs)
            dimension = int(embeddings.shape[-1])
          embeddings = array_ops.gather(
            embeddings, shard_unique_index,
            name='shard_unique_restore')
          embeddings, _ = Collective.get().alltoall(
            embeddings,
            sizes=shard_sizes,
            common_shape=[dimension])
          embeddings = array_ops.gather(
            embeddings, shard_index,
            name='shard_stitch')
        return embeddings

    return _wrapped_embedding_lookup

  def wraps_hierarchical_embedding_lookup(self, fn):
    r'''Rewrites embedding_lookup.
    '''
    def _wrapped_embedding_lookup(params, ids, **kwargs):
      r'''Looks up `ids` in a list of embedding tensors.
      '''
      if not self.issharded(params):
        with Context.scope(sharding=False):
          return fn(params, ids, **kwargs)

      current_device = control_flow_ops.no_op().device
      with Context.scope(sharding=False):
        with ops.device(Context.get().devices[Context.get().rank]):
          s0_ids_shards, s0_ids_sizes, s0_shard_index =\
            partition_by_dual_modulo_stage_one(
              array_ops.reshape(ids, shape=[-1]),
              self._local_world_size, self._num_nodes,
              name='s0_shard_partition')
          s0_shard_ids, s0_shard_sizes = Collective.get().alltoall(
            s0_ids_shards,
            sizes=s0_ids_sizes,
            topology=Topology.INTRA_NODE)

          s0_shard_ids, s0_shard_unique_index = array_ops.unique(
            array_ops.reshape(s0_shard_ids, shape=[-1]),
            name='s0_shard_unique')
          s1_ids_shards, s1_ids_sizes, s1_shard_index =\
            partition_by_dual_modulo_stage_two(
              s0_shard_ids, self._num_nodes, self._local_world_size,
              name='s1_shard_partition')
          s1_shard_ids, s1_shard_sizes = Collective.get().alltoall(
            s1_ids_shards,
            sizes=s1_ids_sizes,
            topology=Topology.INTER_NODE)
          s1_shard_ids, s1_shard_unique_index = array_ops.unique(
            array_ops.reshape(s1_shard_ids, shape=[-1]),
            name='s1_shard_unique')

          if params not in ops.get_collection_ref(GraphKeys.DYNAMIC_VARIABLES):
            s1_shard_ids = s1_shard_ids // Context.get().world_size

          with ops.device(current_device):
            embeddings = fn(params, s1_shard_ids, **kwargs)
            dimension = int(embeddings.shape[-1])

          embeddings = array_ops.gather(
            embeddings, s1_shard_unique_index,
            name='s1_shard_unique_restore')

          embeddings, _ = Collective.get().alltoall(
            embeddings,
            sizes=s1_shard_sizes,
            common_shape=[dimension],
            topology=Topology.INTER_NODE)
          embeddings = array_ops.gather(
            embeddings, s1_shard_index,
            name='s1_shard_stitch')
          embeddings = array_ops.gather(
            embeddings, s0_shard_unique_index,
            name='s0_shard_unique_restore')

          embeddings, _ = Collective.get().alltoall(
            embeddings,
            sizes=s0_shard_sizes,
            common_shape=[dimension],
            topology=Topology.INTRA_NODE)
          embeddings = array_ops.gather(
            embeddings, s0_shard_index,
            name='s0_shard_stitch')
        return embeddings

    return _wrapped_embedding_lookup

  def wraps_shared_embedding_get_dense_tensor_internal(self, fn):
    r'''Rewrites to prevent an incorrect embedding_shape.
    '''
    def _wrapped_shared_embedding_get_dense_tensor_internal(
        cls, *args, **kwargs):
      r'''Get dense tensors in shared embedding column.
      '''
      shared_embedding_collection = ops.get_collection(
        cls.shared_embedding_collection_name)
      if shared_embedding_collection:
        embedding_shape = tensor_shape.TensorShape(
          [tensor_shape.as_dimension(cls.categorical_column._num_buckets),  # pylint: disable=protected-access
           tensor_shape.as_dimension(cls.dimension)])
        embedding_weights = shared_embedding_collection[0]
        prev_embedding_shape = embedding_weights.get_shape
        embedding_weights.get_shape = lambda: embedding_shape
        ret = fn(cls, *args, **kwargs)
        embedding_weights.get_shape = prev_embedding_shape
        return ret
      return fn(cls, *args, **kwargs)
    return _wrapped_shared_embedding_get_dense_tensor_internal

  def wraps_embeddings_from_arguments(self, fn):
    r'''Rewrites to prevent an incorrect embedding_shape.
    '''
    def _wrapped_embeddings_from_arguments(
        column, arg, *args, **kwargs):
      r'''Get dense tensors in shared embedding column.
      '''
      if arg.shared_embedding_name is not None:
        shared_embedding_collection_name = ('SHARED_EMBEDDING_COLLECTION_'
                                            + arg.shared_embedding_name.upper())
        graph = ops.get_default_graph()
        shared_embedding_collection = (
          graph.get_collection_ref(shared_embedding_collection_name))
        if shared_embedding_collection:
          embedding_shape = tensor_shape.TensorShape(
            [tensor_shape.as_dimension(arg.vocab_size),  # pylint: disable=protected-access
             tensor_shape.as_dimension(arg.dimension)])
          embedding_weights = shared_embedding_collection[0]
          prev_embedding_shape = embedding_weights.get_shape
          embedding_weights.get_shape = lambda: embedding_shape
          ret = fn(column, arg, *args, **kwargs)
          embedding_weights.get_shape = prev_embedding_shape
          return ret
        return fn(column, arg, *args, **kwargs)
      return fn(column, arg, *args, **kwargs)
    return _wrapped_embeddings_from_arguments

  def begin(self):
    r'''Rewrites API.
    '''
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    self._prev_lookup = embedding_ops.embedding_lookup

    if (Context.get().options.use_hierarchical_embedding_lookup
        and self._local_world_size > 1
        and self._num_nodes > 1):
      embedding_ops.embedding_lookup = self.wraps_hierarchical_embedding_lookup(
        embedding_ops.embedding_lookup)
      tf.nn.embedding_lookup = self.wraps_hierarchical_embedding_lookup(
        embedding_ops.embedding_lookup)
    else:
      embedding_ops.embedding_lookup = self.wraps_embedding_lookup(
        embedding_ops.embedding_lookup)
      tf.nn.embedding_lookup = self.wraps_embedding_lookup(
        embedding_ops.embedding_lookup)

    # pylint: disable=protected-access
    self._prev_shared_emb_get_dense_tensor_internal = (
      _shared_emb._get_dense_tensor_internal)
    _shared_emb._get_dense_tensor_internal = (
      self.wraps_shared_embedding_get_dense_tensor_internal(
        self._prev_shared_emb_get_dense_tensor_internal))

    # pylint: disable=protected-access
    self._prev_embeddings_from_arguments = (
      _contrib_feature_column._embeddings_from_arguments)
    _contrib_feature_column._embeddings_from_arguments = (
      self.wraps_embeddings_from_arguments(
        self._prev_embeddings_from_arguments))

  def end(self):
    r'''Revert API rewriting.
    '''
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    embedding_ops.embedding_lookup = self._prev_lookup
    tf.nn.embedding_lookup = self._prev_lookup
    _shared_emb._get_dense_tensor_internal = (  # pylint: disable=protected-access
      self._prev_shared_emb_get_dense_tensor_internal)
    _contrib_feature_column._embeddings_from_arguments = (  # pylint: disable=protected-access
      self._prev_embeddings_from_arguments)


GraphRewriting.register(ShardedEmbeddingLookupRewriting)

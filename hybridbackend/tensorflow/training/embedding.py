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

r'''Embedding lookup related classes and functions.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops

from hybridbackend.tensorflow.distribute.embedding import \
  ParallelEmbeddingLookupContext
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys
from hybridbackend.tensorflow.framework.ops import ModeKeys
from hybridbackend.tensorflow.framework.rewriting import GraphRewriting


class EmbeddingLookupRewriting(GraphRewriting):  # pylint: disable=useless-object-inheritance
  r'''Rewriting embedding lookups.
  '''
  @classmethod
  def register(cls, rewriting):
    r'''Register implementation.

    Args:
      rewriting: Implementation class to register.
    '''
    return GraphRewriting.register(rewriting)

  @property
  def should_shard(self):
    r'''Whether embedding weights sharding should be enabled.
    '''
    ctx = Context.get()
    return (
      ctx.world_size > 1
      and ctx.options.sharding
      and ctx.options.mode != ModeKeys.PREDICT)

  @property
  def num_shards(self):
    return Context.get().world_size

  @property
  def shard(self):
    return Context.get().rank

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
      GraphKeys.SHARDED_VARIABLES,
      self.__class__.__name__]
    if collections is None:
      collections = list(minimal_collections)
    else:
      collections.extend(minimal_collections)
      collections = list(set(collections))
    return collections

  def _is_sharded(self, weights):
    r'''Check whether the embedding weights are sharded.
    '''
    sharded_weights_set = ops.get_default_graph().get_collection_ref(
      self.__class__.__name__)
    if weights in sharded_weights_set:
      return True
    if isinstance(weights, (list, tuple)) and len(weights) == 1:
      weights = weights[0]
      if isinstance(weights, ops.Tensor):
        vname = weights.name.split('/read')[0]
        for v in sharded_weights_set:
          if vname == v.name.split(':')[0]:
            return True
    return False

  def wraps_build_embedding_weights(self, fn):
    r'''Rewrites weights building.
    '''
    def _wrapped_build_weights(name, *args, **kwargs):
      r'''Build embedding weights.
      '''
      if not self.should_shard:
        return self.build_unsharded_weights(fn, name, *args, **kwargs)

      collections = kwargs.pop('collections', None)
      kwargs['collections'] = self._sharded_embedding_weights_collections(
        collections)
      return self.build_sharded_weights(
        self.shard, self.num_shards, self._nosharding(fn),
        name, *args, **kwargs)
    return _wrapped_build_weights

  def wraps_embedding_lookup(self, fn):
    r'''Rewrites embedding_lookup.
    '''
    def _wrapped_embedding_lookup(params, ids, **kwargs):
      r'''Looks up `ids` in a list of embedding tensors.
      '''
      if not self._is_sharded(params):
        with Context.scope(sharding=False):
          return fn(params, ids, **kwargs)

      current_device = control_flow_ops.no_op().device
      with Context.scope(sharding=False):
        with ops.device(Context.get().devices[Context.get().rank]):
          ctx = ParallelEmbeddingLookupContext(current_device)
          ids = ctx.exchange_ids(ids)
          ids = self.build_sharded_ids(ids)
          with ops.device(ctx.device):
            embeddings = fn(params, ids, **kwargs)
        return ctx.exchange_embeddings(embeddings)

    return _wrapped_embedding_lookup

  def build_sharded_ids(self, ids):
    r'''Shard IDs to lookup sharded embedding weights.
    '''
    return ids

  def build_unsharded_weights(self, fn, name, *args, **kwargs):
    r'''Build unsharded embedding weights.
    '''
    return fn(name, *args, **kwargs)

  def build_sharded_weights(self, shard, num_shards, fn, name, *args, **kwargs):
    r'''Build sharded embedding weights.
    '''
    del shard
    del num_shards
    return fn(name, *args, **kwargs)

  @abc.abstractmethod
  def begin(self):
    r'''Rewrites API.
    '''

  @abc.abstractmethod
  def end(self):
    r'''Revert API rewriting.
    '''

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

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops

from hybridbackend.tensorflow.distribute.embedding import \
  ParallelEmbeddingLookupContext
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.context import context_scope
from hybridbackend.tensorflow.framework.ops import GraphKeys


class EmbeddingLookupPatching(object):  # pylint: disable=useless-object-inheritance
  r'''Patching embedding lookups.
  '''
  @property
  def name(self):
    r'''Name of the patching.
    '''
    raise NotImplementedError

  @property
  def sharding(self):
    r'''Whether sharding should be enabled.
    '''
    return Context.get().world_size > 1 and Context.get().options.sharding

  @property
  def num_shards(self):
    return Context.get().world_size

  @property
  def shard(self):
    return Context.get().rank

  def call(self, fn, *args, **kwargs):
    r'''Call fn without sharding.
    '''
    with context_scope(sharding=False):
      return fn(*args, **kwargs)

  def wraps(self, fn):
    r'''Wraps fn without sharding.
    '''
    def wrapped_fn(*args, **kwargs):
      with context_scope(sharding=False):
        return fn(*args, **kwargs)
    return wrapped_fn

  def weights_collections(self, collections):
    r'''Build collections for sharded weights.
    '''
    minimal_collections = [
      ops.GraphKeys.GLOBAL_VARIABLES,
      GraphKeys.SHARDED_VARIABLES,
      self.name]
    if collections is None:
      collections = list(minimal_collections)
    else:
      collections.extend(minimal_collections)
      collections = list(set(collections))
    return collections

  @property
  def weights_list(self):
    r'''List of sharded weights.
    '''
    return ops.get_default_graph().get_collection_ref(self.name)

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

  def wraps_build_weights(self, fn):
    r'''Patches weights building.
    '''
    def _wrapped_build_weights(name, *args, **kwargs):
      r'''Build embedding weights.
      '''
      if not self.sharding:
        return self.build_unsharded_weights(fn, name, *args, **kwargs)

      collections = kwargs.pop('collections', None)
      kwargs['collections'] = self.weights_collections(collections)
      return self.build_sharded_weights(
        self.shard, self.num_shards, self.wraps(fn),
        name, *args, **kwargs)
    return _wrapped_build_weights

  def weights_sharded(self, weights):
    r'''Check whether the embedding weights are sharded.
    '''
    _ = weights
    return False

  def shard_ids(self, ids):
    r'''Shard IDs to lookup sharded embedding weights.
    '''
    return ids

  def wraps_embedding_lookup(self, fn):
    r'''Patches embedding_lookup.
    '''
    def _wrapped_embedding_lookup(params, ids, **kwargs):
      r'''Looks up `ids` in a list of embedding tensors.
      '''
      if params not in self.weights_list and not self.weights_sharded(params):
        return self.call(fn, params, ids, **kwargs)

      current_device = control_flow_ops.no_op().device
      with ops.device(Context.get().devices[Context.get().rank]):
        ctx = ParallelEmbeddingLookupContext(current_device)
        ids = ctx.exchange_ids(ids)
        ids = self.shard_ids(ids)
        with ops.device(ctx.device):
          embeddings = self.call(fn, params, ids, **kwargs)
        return ctx.exchange_embeddings(embeddings)

    return _wrapped_embedding_lookup

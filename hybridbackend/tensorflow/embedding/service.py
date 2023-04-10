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

r'''Embedding service on offloaded storage.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import training_util

from hybridbackend.tensorflow.distribute.collective import Collective
from hybridbackend.tensorflow.distribute.ops import Topology
from hybridbackend.tensorflow.distribute.partition.ops import \
  partition_by_dual_modulo_stage_one
from hybridbackend.tensorflow.distribute.partition.ops import \
  partition_by_dual_modulo_stage_two
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys
from hybridbackend.tensorflow.framework.view import OperationLike


class EmbeddingService(
    namedtuple(
      'EmbeddingService', [
        'name', 'params',
        'cache_slab_size', 'cache_size',
        'cache_keys', 'cache_values', 'cache_steps'])):
  r'''Embedding service on offloaded storage.
  '''
  class CacheUpdates(object):  # pylint: disable=useless-object-inheritance
    r'''Updates to embedding table cache.
    '''
    def __init__(self):
      self._removed_cache_keys = []
      self._removed_cache_indices = []
      self._added_cache_keys = []
      self._added_cache_indices = []

    def remove(self, key, cache_indices):
      self._removed_cache_keys.append(key)
      self._removed_cache_indices.append(cache_indices)

    def add(self, key, cache_indices):
      self._added_cache_keys.append(key)
      self._added_cache_indices.append(cache_indices)

    def all_removed_cache_indices(self):
      return array_ops.concat(self._removed_cache_indices, axis=0)

    def all_removed_cache_keys(self):
      return array_ops.concat(self._removed_cache_keys, axis=0)

    def all_added_cache_indices(self):
      return array_ops.concat(self._added_cache_indices, axis=0)

    def all_added_cache_keys(self):
      return array_ops.concat(self._added_cache_keys, axis=0)

  EMPTY = - 2 ** 63

  def __new__(cls, capacity, builder, *args, collections=None, **kwargs):
    params_collections = [
      ops.GraphKeys.GLOBAL_VARIABLES,
      GraphKeys.NOT_REPLICATED]
    if collections is not None:
      params_collections = list(set(collections + params_collections))
    params = builder(*args, collections=params_collections, **kwargs)
    name = params.name.split(':')[0]
    cache_slab_size = 32
    try:
      import pycuda.autoinit  # pylint: disable=import-outside-toplevel
      cache_slab_size = pycuda.autoinit.device.get_attribute(
        pycuda.driver.device_attribute.WARP_SIZE)
    except:  # pylint: disable=bare-except
      pass

    capacity = (capacity // cache_slab_size) * cache_slab_size
    cache_size = vs.get_variable(
      f'{name}_cache_size',
      shape=[],
      dtype=dtypes.int32,
      collections=[ops.GraphKeys.LOCAL_VARIABLES, GraphKeys.NOT_REPLICATED],
      initializer=init_ops.constant_initializer(capacity),
      trainable=False,
      use_resource=True)
    cache_keys = vs.get_variable(
      f'{name}_cache_keys',
      shape=[capacity],
      dtype=dtypes.int64,
      collections=[ops.GraphKeys.LOCAL_VARIABLES, GraphKeys.NOT_REPLICATED],
      initializer=init_ops.constant_initializer(EmbeddingService.EMPTY),
      trainable=False,
      use_resource=True)
    cache_values = vs.get_variable(
      f'{name}_cache_values',
      shape=[capacity, params.shape[-1]],
      dtype=params.dtype,
      collections=[ops.GraphKeys.LOCAL_VARIABLES, GraphKeys.NOT_REPLICATED],
      initializer=init_ops.zeros_initializer(),
      trainable=True,
      use_resource=True)
    cache_steps = vs.get_variable(
      f'{name}_cache_steps',
      shape=[capacity],
      dtype=dtypes.int64,
      collections=[ops.GraphKeys.LOCAL_VARIABLES, GraphKeys.NOT_REPLICATED],
      initializer=init_ops.constant_initializer(EmbeddingService.EMPTY),
      trainable=False,
      use_resource=True)
    return super(EmbeddingService, cls).__new__(
      cls, name, params,
      cache_slab_size, cache_size,
      cache_keys, cache_values, cache_steps)

  @abc.abstractmethod
  def pull(self, storage, keys):
    r'''Pull embeddings from storage.
    '''

  @abc.abstractmethod
  def push(self, storage, keys, values):
    r'''Push values to storage.
    '''

  def lookup(self, ids, out_cache_updates):
    r'''Lookup embeddings from parameters via caching.
    '''
    @custom_gradient.custom_gradient
    def _lookup_op(cache, keys):
      hit_input_indices, hit_cache_indices, miss_input_indices, miss_keys = (
        OperationLike('Lookup')
        .returns_tensors(
          tensor_spec.TensorSpec([None], dtypes.int32),
          tensor_spec.TensorSpec([None], dtypes.int32),
          tensor_spec.TensorSpec([None], dtypes.int32),
          tensor_spec.TensorSpec([None], dtypes.int64))
        .finalize(self.cache_keys, keys, cache_slab_size=self.cache_slab_size))
      hit_values = array_ops.gather(cache, hit_cache_indices)
      values = array_ops.zeros(
        keys.shape.concatenate(self.params.shape[-1]), self.params.dtype)
      values = array_ops.tensor_scatter_update(
        values, hit_input_indices, hit_values)
      miss_values = self.pull(self.params, miss_keys)
      values = array_ops.tensor_scatter_update(
        values, miss_input_indices, miss_values)

      def grad_fn(*grads):
        r'''Gradient function for embedding lookup.
        '''
        num_misses = array_ops.size(miss_keys)

        def _evict():
          step = training_util.get_or_create_global_step()
          num_removes = num_misses - self.cache_size
          hit_steps_updated = state_ops.scatter_update(
            self.cache_steps, hit_cache_indices, -step)
          with ops.control_dependencies([hit_steps_updated]):
            _, removed_cache_indices = nn_ops.top_k(
              self.cache_steps, num_removes, sorted=False)
          steps_removed = state_ops.scatter_update(
            self.cache_steps, removed_cache_indices, EmbeddingService.EMPTY)
          removed_keys = array_ops.gather(
            self.cache_keys, removed_cache_indices)
          out_cache_updates.remove(removed_keys, removed_cache_indices)
          with ops.control_dependencies([removed_keys]):
            keys_removed = state_ops.scatter_update(
              self.cache_keys, removed_cache_indices, EmbeddingService.EMPTY)
          removed_cache_values = array_ops.gather(
            self.cache_values, removed_cache_indices)
          params_updated = self.push(
            self.params, removed_keys, removed_cache_values)
          with ops.control_dependencies(
              [steps_removed, keys_removed, params_updated]):
            return self.cache_size.assign(0)

        def _inact():
          return self.cache_size.assign_sub(num_misses)

        # TODO: Pack and linearize removal ops
        with ops.control_dependencies([values]):
          remove_ready = control_flow_ops.cond(
            num_misses > self.cache_size, _evict, _inact)
        with ops.control_dependencies([remove_ready]):
          available_cache_indices = array_ops.where(
            math_ops.equal(self.cache_keys, EmbeddingService.EMPTY))
          miss_cache_indices = array_ops.slice(
            available_cache_indices, 0, num_misses)
          reserve_ready = state_ops.scatter_update(
            cache, miss_cache_indices, miss_values)
        out_cache_updates.add(miss_keys, miss_cache_indices)
        cache_indices = array_ops.zeros(keys.shape, dtypes.int32)
        cache_indices = array_ops.tensor_scatter_update(
          cache_indices, hit_input_indices, hit_cache_indices)
        cache_indices = array_ops.tensor_scatter_update(
          cache_indices, miss_input_indices, miss_cache_indices)
        with ops.colocate_with(cache):
          cache_shape = array_ops.shape(cache)
        with ops.control_dependencies([reserve_ready]):
          d_cache = ops.IndexedSlices(
            array_ops.identity(grads[0]), cache_indices, cache_shape)
        return d_cache, None
      return values, grad_fn

    num_devices_per_node = Context.get().local_world_size
    num_nodes = Context.get().world_size // num_devices_per_node

    s0_ids_shards, s0_ids_sizes, s0_shard_index =\
      partition_by_dual_modulo_stage_one(
        array_ops.reshape(ids, shape=[-1]),
        num_devices_per_node, num_nodes,
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
        s0_shard_ids, num_nodes, num_devices_per_node,
        name='s1_shard_partition')
    s1_shard_ids, s1_shard_sizes = Collective.get().alltoall(
      s1_ids_shards,
      sizes=s1_ids_sizes,
      topology=Topology.INTER_NODE)
    s1_shard_ids, s1_shard_unique_index = array_ops.unique(
      array_ops.reshape(s1_shard_ids, shape=[-1]),
      name='s1_shard_unique')
    s1_shard_ids = s1_shard_ids // Context.get().world_size
    embeddings = _lookup_op(self.cache_values, s1_shard_ids)
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

  def before_apply_gradients(
      self, optimizer, cache_updates, out_optimizer_states):
    r'''Update optimizer states before apply.
    '''
    update_ops = []
    for state_name in optimizer.get_slot_names():
      states = optimizer.get_slot(self.params, state_name)
      states_cache = optimizer.get_slot(self.cache_values, state_name)
      out_optimizer_states.append((states, states_cache))

      states_to_remove = array_ops.gather(
        states_cache,
        cache_updates.all_removed_cache_indices())
      states_removed = self.push(
        states,
        cache_updates.all_removed_cache_keys(),
        states_to_remove)
      with ops.control_dependencies([states_removed]):
        states_to_add = self.pull(
          states,
          cache_updates.all_added_cache_keys())
      states_added = state_ops.scatter_update(
        states_cache,
        cache_updates.all_added_cache_indices(),
        states_to_add)
      update_ops.append(states_added)
    return array_ops.group(update_ops)

  def before_save_checkpoints(self, optimizer_states):
    r'''Update parameters and optimizer states before checkpointing.
    '''
    all_cache_indices = array_ops.where(
      math_ops.not_equal(self.cache_keys, EmbeddingService.EMPTY))
    all_cache_keys = array_ops.gather(self.cache_keys, all_cache_indices)
    all_cache_values = array_ops.gather(self.cache_values, all_cache_indices)
    update_ops = [self.push(self.params, all_cache_keys, all_cache_values)]
    for states, cache_states in enumerate(optimizer_states):
      all_cache_states = array_ops.gather(cache_states, all_cache_indices)
      update_ops.append(self.push(states, all_cache_keys, all_cache_states))
    return array_ops.group(update_ops)

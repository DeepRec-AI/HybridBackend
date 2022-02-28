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

r'''Functors for coalesced embedding lookup.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients

from hybridbackend.tensorflow.distribute.communicator_pool import \
    CommunicatorPool
from hybridbackend.tensorflow.feature_column.embedding_backend import \
    EmbeddingBackend
from hybridbackend.tensorflow.feature_column.embedding_lookup_buffered import \
    EmbeddingLookupBuffered
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.ops.math_ops import floormod_partition
from hybridbackend.tensorflow.ops.math_ops import segment_combine


class EmbeddingLookupCoalesced(object): # pylint: disable=useless-object-inheritance
  r'''Functor to lookup embeddings in group.
  '''
  def __init__(self):
    self._impl = EmbeddingBackend.get()

  def _exchange_group_ids(self, columns):
    r'''Communicator call to exchange group IDs.
    '''
    comm_size = len(Context.get().devices)
    def comm_fn(comm, inputs, inputs_deps):
      r'''Wrapped function with communicator.
      '''
      group_ids = [
          inputs[i*comm_size:(i+1)*comm_size]
          for i, _ in enumerate(columns)]
      with ops.control_dependencies(inputs_deps):
        group_mp_ids = comm.group_alltoallw(group_ids)
        flatten_group_mp_ids = sum(group_mp_ids, [])
        return flatten_group_mp_ids, None
    return comm_fn

  def _exchange_group_embeddings(self, columns):
    r'''Communicator call to exchange group embeddings.
    '''
    comm_size = len(Context.get().devices)
    def comm_fn(comm, inputs, inputs_deps):
      r'''Wrapped function with communicator.
      '''
      group_mp_embs = [
          inputs[i*comm_size:(i+1)*comm_size]
          for i, _ in enumerate(columns)]
      if inputs_deps is None:
        group_embs = comm.group_alltoallw(group_mp_embs)
        flatten_group_embs = sum(group_embs, [])
        return flatten_group_embs, None
      with ops.control_dependencies(inputs_deps):
        group_embs = comm.group_alltoallw(group_mp_embs)
      flatten_group_embs = sum(group_embs, [])
      def grad_fn(grads, grads_deps, roots):
        r'''Gradient function.
        '''
        del roots
        with ops.control_dependencies(grads_deps):
          d_inputs = gradients.gradients(
              flatten_group_embs, inputs, grad_ys=grads)
        return d_inputs, None
      return flatten_group_embs, grad_fn
    return comm_fn

  def __call__(
      self, group_weights, group_inputs, columns, num_groups=1, name=None):
    r'''Lookup embedding results.
    '''
    # Transform inputs.
    group_sparse_ids = []
    group_ids_shards = []
    group_partition_index = []
    group_sparse_weights = []
    group_unique_index = []
    with ops.name_scope('transform_ids'):
      for idx, column in enumerate(columns):
        with ops.name_scope(column.name):
          if not self._impl.sharded(column):
            raise ValueError(
                'Only sharded embedding columns can use group call.')
          sparse_ids = group_inputs[idx].id_tensor
          sparse_weights = group_inputs[idx].weight_tensor
          if isinstance(sparse_ids, sparse_tensor.SparseTensor):
            ids = sparse_ids.values
          else:
            ids = ops.convert_to_tensor(sparse_ids)
          ids = array_ops.reshape(ids, [-1])
          if self._impl.unique(column):
            unique_index = None
          else:
            ids, unique_index = array_ops.unique(ids)
          with ops.device(self._impl.op_device(column)):
            ids_shards, partition_index = floormod_partition(
                ids, len(Context.get().devices))
          group_sparse_ids.append(sparse_ids)
          group_ids_shards.append(list(ids_shards))
          group_partition_index.append(partition_index)
          group_sparse_weights.append(sparse_weights)
          group_unique_index.append(unique_index)

    # Split into buckets.
    group_size = len(columns)
    num_groups = min(num_groups, group_size)
    bucket_size = group_size // num_groups
    bucket_sizes = [bucket_size for _ in range(num_groups)]
    for r in range(group_size - num_groups * bucket_size):
      bucket_sizes[r] += 1

    bucket_offset = 0
    bucket_columns = []
    bucket_ids_shards = []
    bucket_partition_index = []
    bucket_sparse_ids = []
    bucket_sparse_weights = []
    bucket_unique_index = []
    bucket_weights = []
    for bucket_size in bucket_sizes:
      bucket_columns.append(
          columns[bucket_offset: bucket_offset + bucket_size])
      bucket_ids_shards.append(
          group_ids_shards[bucket_offset: bucket_offset + bucket_size])
      bucket_partition_index.append(
          group_partition_index[bucket_offset: bucket_offset + bucket_size])
      bucket_sparse_ids.append(
          group_sparse_ids[bucket_offset: bucket_offset + bucket_size])
      bucket_sparse_weights.append(
          group_sparse_weights[bucket_offset: bucket_offset + bucket_size])
      bucket_unique_index.append(
          group_unique_index[bucket_offset: bucket_offset + bucket_size])
      bucket_weights.append(
          group_weights[bucket_offset: bucket_offset + bucket_size])
      bucket_offset += bucket_size

    # Exchange IDs in buckets.
    comm_size = len(Context.get().devices)
    bucket_mp_ids_shards = []
    with ops.name_scope('exchange_ids'):
      for bid, bucket_size in enumerate(bucket_sizes):
        flatten_bucket_ids_shards = sum(bucket_ids_shards[bid], [])
        flatten_bucket_mp_ids_shards = CommunicatorPool.get().call(
            self._exchange_group_ids(bucket_columns[bid]),
            flatten_bucket_ids_shards,
            trainable=False)
        bucket_mp_ids_shards.append([
            flatten_bucket_mp_ids_shards[i*comm_size:(i+1)*comm_size]
            for i, _ in enumerate(bucket_columns[bid])])

    # Lookup embeddings.
    bucket_mp_embs_shards = []
    with ops.name_scope('lookup_embeddings'):
      lookup_buffered = EmbeddingLookupBuffered(bucket_columns)
      bucket_mp_embs_shards = lookup_buffered(
          bucket_weights, bucket_mp_ids_shards)

    # Exchange embeddings.
    bucket_embs_shards = []
    with ops.name_scope('exchange_embeddings'):
      for bid, bucket_size in enumerate(bucket_sizes):
        mp_embs_shards = [
            [array_ops.reshape(t, [-1]) for t in embedding_shards]
            for embedding_shards in bucket_mp_embs_shards[bid]]
        flatten_bucket_mp_embs_shards = sum(mp_embs_shards, [])
        flatten_bucket_embs_shards = CommunicatorPool.get().call(
            self._exchange_group_embeddings(bucket_columns[bid]),
            flatten_bucket_mp_embs_shards)
        bucket_embs_shards.append([
            flatten_bucket_embs_shards[i*comm_size:(i+1)*comm_size]
            for i, _ in enumerate(bucket_columns[bid])])

    # Transform embeddings.
    group_results = []
    with ops.name_scope('transform_embeddings'):
      for bid, bucket_size in enumerate(bucket_sizes):
        for idx, column in enumerate(bucket_columns[bid]):
          dimension = self._impl.dimension(column)
          with ops.name_scope(column.name):
            embeddings_shards = [
                array_ops.reshape(s, [-1, dimension])
                for s in bucket_embs_shards[bid][idx]]
            with ops.device(self._impl.op_device(column)):
              embeddings = data_flow_ops.dynamic_stitch(
                  bucket_partition_index[bid][idx], embeddings_shards)
            group_results.append(
                segment_combine(
                    bucket_sparse_ids[bid][idx], embeddings,
                    weights=bucket_sparse_weights[bid][idx],
                    indices=bucket_unique_index[bid][idx],
                    dimension=dimension,
                    pad=self._impl.pad(column),
                    segment_rank=self._impl.segment_rank(column),
                    combiner=column.combiner,
                    name=name))
    return group_results

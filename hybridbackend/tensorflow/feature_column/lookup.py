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

r'''Functors for embedding lookup.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.platform import tf_logging as logging

from hybridbackend.tensorflow.distribute.communicator_pool import \
  CommunicatorPool
from hybridbackend.tensorflow.feature_column.backend import EmbeddingBackend
from hybridbackend.tensorflow.feature_column.math_lib import segment_reduce
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.ops.floormod_shuffle.ops import floormod_shuffle
from hybridbackend.tensorflow.ops.floormod_shuffle.ops import \
  floormod_shuffle_n


class EmbeddingLookup(object):  # pylint: disable=useless-object-inheritance
  r'''Functor to lookup embeddings.
  '''
  def __init__(self):
    self._impl = EmbeddingBackend.get()

  def _exchange_ids(self, comm, inputs, inputs_deps):
    r'''Communicator call to exchange IDs.
    '''
    with ops.control_dependencies(inputs_deps):
      return comm.alltoallv(*inputs), None

  def _exchange_embeddings(self, dimension):
    r'''Communicator call to exchange embeddings.
    '''
    def _fn(comm, inputs, inputs_deps):
      r'''Communicator call to exchange embeddings.
      '''
      if inputs_deps is None:
        embs, _ = comm.alltoallv(*inputs, common_shape=[dimension])
        return embs, None
      with ops.control_dependencies(inputs_deps):
        embs, _ = comm.alltoallv(*inputs, common_shape=[dimension])

      def grad_fn(grads, grads_deps, roots):
        r'''Gradient function.
        '''
        del roots
        with ops.control_dependencies(grads_deps):
          d_inputs = gradients.gradients(embs, inputs, grad_ys=grads)
        return d_inputs, None
      return embs, grad_fn
    return _fn

  def __call__(self, column, weights, inputs, name=None):
    r'''Lookup embedding results for sparse tensors.
    '''
    with ops.name_scope(name):
      sparse_ids = inputs.id_tensor
      sparse_weights = inputs.weight_tensor
      if isinstance(sparse_ids, sparse_tensor.SparseTensor):
        ids = sparse_ids.values
      else:
        ids = sparse_ids
      input_device = self._impl.input_device(column)
      pad = self._impl.pad(column)
      segment_rank = self._impl.segment_rank(column)
      with ops.device(input_device):
        ids = array_ops.reshape(ops.convert_to_tensor(ids), [-1])
        if self._impl.unique(column):
          unique_index = None
          if not pad:
            logging.info('If unique is set to True, pad would be set to True')
            pad = True
        else:
          ids, unique_index = array_ops.unique(ids)
        if segment_rank != 0 and not pad:
          logging.info('If segment rank is not 0, pad would be set to True')
          pad = True
      dimension = self._impl.dimension(column)
      if self._impl.sharded(column):
        with ops.name_scope('shuffle_ids'):
          ids_shards, ids_sizes, partition_index = floormod_shuffle(
            ids, Context.get().world_size)
          shard_ids, embs_sizes = CommunicatorPool.get().alltoallv(
            ids_shards, ids_sizes,
            trainable=False)
          shard_ids, shard_unique_index = array_ops.unique(shard_ids)
        with ops.device(self._impl.device(column)):
          with Context.scope(sharding=False):
            shard_embs = self._impl.lookup(
              column, weights, shard_ids, sharded=True)
        with ops.name_scope('shuffle_embeddings'):
          if shard_unique_index is not None:
            shard_embs = array_ops.gather(
              shard_embs, shard_unique_index, name='restore_unique')
          embs_shards = CommunicatorPool.get().call(
            self._exchange_embeddings(dimension), [shard_embs, embs_sizes])
          embeddings = array_ops.gather(
            embs_shards, partition_index, name='restore_shuffle')
      else:
        with ops.device(self._impl.device(column)):
          with Context.scope(sharding=False):
            embeddings = self._impl.lookup(column, weights, ids)
      return segment_reduce(
        sparse_ids, embeddings,
        weights=sparse_weights,
        indices=unique_index,
        dimension=dimension,
        pad=pad,
        segment_rank=segment_rank,
        combiner=self._impl.combiner(column))


class EmbeddingLookupCoalesced(object):  # pylint: disable=useless-object-inheritance
  r'''Functor to lookup embeddings in group.
  '''
  def __init__(self):
    self._impl = EmbeddingBackend.get()

  def _exchange_group_ids(self, comm, inputs, inputs_deps):
    r'''Communicator call to exchange group IDs.
    '''
    num_columns = len(inputs) // 2
    if len(inputs) != 2 * num_columns:
      raise ValueError('Number of IDs to exchange is wrong')
    values = inputs[:num_columns]
    sizes = inputs[num_columns:]
    with ops.control_dependencies(inputs_deps):
      return comm.alltoallv_n(values, sizes), None

  def _exchange_group_embeddings(self, wire_dtype, common_shapes):
    r'''Communicator call to exchange group embeddings.
    '''
    def _fn(comm, inputs, inputs_deps):
      num_columns = len(inputs) // 2
      if len(inputs) != 2 * num_columns:
        raise ValueError('Number of IDs to exchange is wrong')
      values = inputs[:num_columns]
      sizes = inputs[num_columns:]
      if inputs_deps is None:
        embs, _ = comm.alltoallv_n(
          values, sizes,
          wire_dtype=wire_dtype,
          common_shapes=common_shapes)
        return embs, None
      with ops.control_dependencies(inputs_deps):
        embs, _ = comm.alltoallv_n(
          values, sizes,
          wire_dtype=wire_dtype,
          common_shapes=common_shapes)

      def grad_fn(grads, grads_deps, roots):
        r'''Gradient function.
        '''
        del roots
        with ops.control_dependencies(grads_deps):
          d_inputs = gradients.gradients(embs, inputs, grad_ys=grads)
        return d_inputs, None
      return embs, grad_fn
    return _fn

  def __call__(
      self, group_weights, group_inputs, columns, num_groups=1, name=None):
    r'''Lookup embedding results.
    '''
    if name is None:
      name = ops.get_default_graph().unique_name('embedding_lookup_coalesced')
    num_shards = len(Context.get().devices)
    num_columns = len(columns)
    num_groups = min(num_groups, num_columns)
    bucket_size = num_columns // num_groups
    bucket_sizes = [bucket_size for _ in range(num_groups)]
    for r in range(num_columns - num_groups * bucket_size):
      bucket_sizes[r] += 1
    bi = 0
    bidx = []
    for bucket_size in bucket_sizes:
      bidx.append(bi)
      bi += bucket_size
    bidx.append(bi)

    # Lookup embeddings in buckets.
    group_embeddings = []
    for bid, _ in enumerate(bucket_sizes):
      with ops.name_scope(f'{name}/bucket{bid}'):
        bucket_columns = columns[bidx[bid]: bidx[bid + 1]]
        bucket_sparse_ids = []
        bucket_sparse_weights = []
        bucket_ids = []
        bucket_unique_index = []
        bucket_pad = []
        with ops.name_scope('unique_ids'):
          for idx, c in enumerate(bucket_columns):
            inputs = group_inputs[bidx[bid]: bidx[bid + 1]][idx]
            sparse_ids = inputs.id_tensor
            bucket_sparse_ids.append(sparse_ids)
            sparse_weights = inputs.weight_tensor
            bucket_sparse_weights.append(sparse_weights)
            if isinstance(sparse_ids, sparse_tensor.SparseTensor):
              ids = sparse_ids.values
            else:
              ids = sparse_ids
            with ops.device(self._impl.input_device(c)):
              ids = array_ops.reshape(ids, [-1])
              bucket_pad.append(self._impl.pad(c))
              if self._impl.unique(c):
                unique_index = None
                if not bucket_pad[idx]:
                  logging.info(
                    'If unique is set to True, pad would be set to True')
                  bucket_pad[idx] = True
              else:
                ids, unique_index = array_ops.unique(ids)
              if self._impl.segment_rank(c) and not bucket_pad[idx]:
                logging.info(
                  'If segment rank is not 0, pad would be set to True')
                bucket_pad[idx] = True
              bucket_ids.append(ids)
              bucket_unique_index.append(unique_index)

        bucket_ids_shards, bucket_ids_sizes, bucket_partition_index = (
          floormod_shuffle_n(bucket_ids, num_shards))
        bucket_shard_ids, bucket_embs_sizes = CommunicatorPool.get().call(
          self._exchange_group_ids,
          bucket_ids_shards + bucket_ids_sizes,
          trainable=False)

        bucket_weights = group_weights[bidx[bid]: bidx[bid + 1]]
        bucket_shard_embs = []
        with ops.name_scope('unique_embs'):
          for idx, c in enumerate(bucket_columns):
            shard_ids = bucket_shard_ids[idx]
            shard_ids, shard_unique_index = array_ops.unique(shard_ids)
            with ops.device(self._impl.device(c)):
              with Context.scope(sharding=False):
                shard_embs = self._impl.lookup(
                  c, bucket_weights[idx], shard_ids,
                  sharded=self._impl.sharded(c))
            if shard_unique_index is not None:
              shard_embs = array_ops.gather(
                shard_embs, shard_unique_index, name='restore_unique')
            bucket_shard_embs.append(shard_embs)

        common_shapes = [[self._impl.dimension(c)] for c in bucket_columns]
        wire_dtypes = {self._impl.wire_dtype(c) for c in bucket_columns}
        if len(wire_dtypes) > 1:
          raise ValueError(
            f'Multiple wire data type ({wire_dtypes}) not supported '
            'in coalescing mode, use consistent `wire_dtype` or '
            'set `num_groups` to `None`')
        bucket_embs_shards = CommunicatorPool.get().call(
          self._exchange_group_embeddings(list(wire_dtypes)[0], common_shapes),
          bucket_shard_embs + bucket_embs_sizes)

        with ops.name_scope('segment_reduce'):
          for idx, c in enumerate(bucket_columns):
            embeddings = array_ops.gather(
              bucket_embs_shards[idx], bucket_partition_index[idx],
              name='restore_shuffle')
            embeddings = segment_reduce(
              bucket_sparse_ids[idx], embeddings,
              weights=bucket_sparse_weights[idx],
              indices=bucket_unique_index[idx],
              dimension=self._impl.dimension(c),
              pad=bucket_pad[idx],
              segment_rank=self._impl.segment_rank(c),
              combiner=self._impl.combiner(c))
            group_embeddings.append(embeddings)
    return group_embeddings

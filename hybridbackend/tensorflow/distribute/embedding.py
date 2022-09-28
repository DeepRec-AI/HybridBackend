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

r'''Butterfly shuffle for embedding lookup.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients

from hybridbackend.tensorflow.distribute.communicator_pool import \
  CommunicatorPool
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.ops.floormod_shuffle.ops import floormod_shuffle


class ParallelEmbeddingLookupContext(object):  # pylint: disable=useless-object-inheritance
  r'''Parallel embedding lookup context.
  '''
  def __init__(self, device):
    self._shard_sizes = None
    self._shard_unique_index = None
    self._shard_index = None
    self._device = device

  @property
  def device(self):
    return self._device

  def exchange_ids(self, ids):
    r'''Exchange IDs across shards.
    '''
    num_shards = Context.get().world_size
    ids_shards, ids_sizes, self._shard_index = floormod_shuffle(
      ids, num_shards, name='shard_partition')
    shard_ids, self._shard_sizes = CommunicatorPool.get().alltoallv(
      ids_shards, ids_sizes, trainable=False)
    shard_ids, self._shard_unique_index = array_ops.unique(
      shard_ids, name='shard_unique')
    return shard_ids

  def exchange_embeddings(self, embeddings):
    r'''Exchange embeddings across shards.
    '''
    def _comm_call_exchange_embeddings(dimension):
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
    dimension = int(embeddings.shape[-1])
    embeddings = array_ops.gather(
      embeddings, self._shard_unique_index,
      name='shard_unique_restore')
    embeddings = CommunicatorPool.get().call(
      _comm_call_exchange_embeddings(dimension),
      [embeddings, self._shard_sizes])
    embeddings = array_ops.gather(
      embeddings, self._shard_index,
      name='shard_stitch')
    return embeddings

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
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients

from hybridbackend.tensorflow.distribute.communicator_pool import \
    CommunicatorPool
from hybridbackend.tensorflow.feature_column.embedding_backend import \
    EmbeddingBackend
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.ops.math_ops import floormod_partition
from hybridbackend.tensorflow.ops.math_ops import segment_combine


class EmbeddingLookup(object): # pylint: disable=useless-object-inheritance
  r'''Functor to lookup embeddings.
  '''
  def __init__(self, column):
    self._column = column
    self._impl = EmbeddingBackend.get()

  def _exchange_ids(self, comm, inputs, inputs_deps):
    r'''Communicator call to exchange IDs.
    '''
    with ops.control_dependencies(inputs_deps):
      return comm.alltoallw(inputs), None

  def _exchange_embeddings(self, comm, inputs, inputs_deps):
    r'''Communicator call to exchange embeddings.
    '''
    if inputs_deps is None:
      embs = comm.alltoallw(inputs, common_shape=[self._column.dimension])
      return embs, None
    with ops.control_dependencies(inputs_deps):
      embs = comm.alltoallw(inputs, common_shape=[self._column.dimension])
    def grad_fn(grads, grads_deps, roots):
      r'''Gradient function.
      '''
      del roots
      with ops.control_dependencies(grads_deps):
        d_inputs = gradients.gradients(embs, inputs, grad_ys=grads)
      return d_inputs, None
    return embs, grad_fn

  def __call__(self, weights, inputs, name=None):
    r'''Lookup embedding results for sparse tensors.
    '''
    sparse_ids = inputs.id_tensor
    sparse_weights = inputs.weight_tensor
    if isinstance(sparse_ids, sparse_tensor.SparseTensor):
      ids = sparse_ids.values
    else:
      ids = ops.convert_to_tensor(sparse_ids)
    ids = array_ops.reshape(ids, [-1])
    if self._impl.unique(self._column):
      unique_index = None
    else:
      ids, unique_index = array_ops.unique(ids)
    if self._impl.sharded(self._column):
      with ops.device(self._impl.op_device(self._column)):
        ids_shards, partition_index = floormod_partition(
            ids, len(Context.get().devices))
      mp_ids_shards = CommunicatorPool.get().call(
          self._exchange_ids, ids_shards,
          trainable=False)
      with ops.device(self._impl.device(self._column)):
        mp_embs_shards = []
        for shard, _ in enumerate(Context.get().devices):
          mp_embs_shards.append(
              self._impl.lookup(weights, mp_ids_shards[shard]))
      embs_shards = CommunicatorPool.get().call(
          self._exchange_embeddings, mp_embs_shards)
      with ops.device(self._impl.op_device(self._column)):
        embeddings = data_flow_ops.dynamic_stitch(
            partition_index, embs_shards)
    else:
      with ops.device(self._impl.device(self._column)):
        embeddings = self._impl.lookup(weights, ids)
    return segment_combine(
        sparse_ids, embeddings,
        weights=sparse_weights,
        indices=unique_index,
        dimension=self._column.dimension,
        pad=self._impl.pad(self._column),
        segment_rank=self._impl.segment_rank(self._column),
        combiner=self._column.combiner,
        name=name)

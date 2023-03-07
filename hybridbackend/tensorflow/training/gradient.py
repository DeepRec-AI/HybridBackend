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

r'''Aggregation of gradients.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

from hybridbackend.tensorflow.distribute.collective import Collective
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys


def _tensor_size(tensor):
  r'''Calculate size of specific tensor.
  '''
  num_elements = tensor.shape.num_elements()
  if num_elements is not None and num_elements > 0:
    return tensor.dtype.size * num_elements
  return -1


def _log(
    dense_vars, dense_grads,
    sparse_vars, sparse_grads,
    sharded_vars, sharded_grads):
  r'''Log values of gradients.
  Args:
    dense_vars: list of dense variables.
    dense_grads: list of dense gradients.
    sparse_vars: list of sparse variables.
    sparse_grads: list of sparse gradients.
    sharded_vars: list of sharded variables.
    sharded_grads: list of sharded gradients.
  '''
  sizes = [_tensor_size(t) for t in dense_grads]
  nonempty_sizes = [s for s in sizes if s > 0]
  sparse_sizes = [
    _tensor_size(s.values) + _tensor_size(s.indices)
    for s in sparse_grads]
  nonempty_sparse_sizes = [s for s in sparse_sizes if s > 0]
  grad_mbs = sum(nonempty_sizes) / 1024.0 / 1024.0
  sparse_grad_mbs = sum(nonempty_sparse_sizes) / 1024.0 / 1024.0
  logging.info(
    f'Aggregate {len(sizes)} dense gradients '
    f'({">" if len(sizes) > len(nonempty_sizes) else ""}{grad_mbs:.2f}MB) '
    f'and {len(sparse_sizes)} sparse gradients '
    f'({">" if len(sparse_sizes) > len(nonempty_sparse_sizes) else ""}'
    f'{sparse_grad_mbs:.2f}MB) in optimizer, '
    f'skip {len(sharded_grads)} gradients')
  for v in dense_vars:
    logging.info(f'Replicated weights: {v.name.split(":")[0]}')
  for v in sparse_vars:
    logging.info(f'Replicated weights: *{v.name.split(":")[0]}')
  for v in sharded_vars:
    logging.info(f'Sharded weights: {v.name.split(":")[0]}')


def _mean(grads, devices):
  r'''Calculate mean gradients.
  '''
  if grads is None or (isinstance(grads, (list, tuple)) and not grads):
    return []
  if (len(devices)) <= 1:
    return grads
  multiplier = 1. / len(devices)
  if grads is None:
    return None
  if isinstance(grads, list):
    return [_mean(g, devices) for g in grads]
  if isinstance(grads, tuple):
    return tuple(_mean(g, devices) for g in grads)
  if isinstance(grads, ops.Tensor):
    multiplier = math_ops.cast(multiplier, grads.dtype.base_dtype)
    return grads * multiplier
  if isinstance(grads, ops.IndexedSlices):
    return ops.IndexedSlices(
      _mean(grads.values, devices),
      grads.indices,
      grads.dense_shape)
  raise ValueError(f'Type of {grads} is not supported')


def _pack_grads_and_vars(
    replica_grads, shard_grads,
    replica_vars, shard_vars,
    replica_grad_indices, shard_grad_indices):
  r'''Packing grads and vars.
  '''
  indexed_grads_and_vars = []
  for i, g in enumerate(shard_grads):
    indexed_grads_and_vars.append(
      (shard_grad_indices[i], (g, shard_vars[i])))
  for i, g in enumerate(replica_grads):
    indexed_grads_and_vars.append(
      (replica_grad_indices[i], (g, replica_vars[i])))
  _, grads_and_vars = zip(*sorted(indexed_grads_and_vars))
  return grads_and_vars


def _aggregate(replicated_vars, replicated_grads, sharded_vars, sharded_grads):
  r'''Aggregate gradients.
  '''
  if not replicated_grads:
    return []

  dense_vars = []
  dense_grads = []
  dense_grad_indices = []
  sparse_vars = []
  sparse_grads = []
  sparse_grad_indices = []
  for idx, g in enumerate(replicated_grads):
    small_weights = ops.get_collection_ref(GraphKeys.TRAINABLE_REPLICATED_SMALL)
    if (isinstance(g, ops.IndexedSlices)
        and replicated_vars[idx] not in small_weights):
      sparse_vars.append(replicated_vars[idx])
      sparse_grads.append(g)
      sparse_grad_indices.append(idx)
    else:
      dense_vars.append(replicated_vars[idx])
      dense_grads.append(ops.convert_to_tensor(g))
      dense_grad_indices.append(idx)
  _log(
    dense_vars, dense_grads,
    sparse_vars, sparse_grads,
    sharded_vars, sharded_grads)
  dense_aggregated = _aggregate_dense(dense_grads)
  sparse_aggregated = _aggregate_sparse(sparse_grads)
  indexed_aggregated = []
  for i, g in enumerate(dense_aggregated):
    indexed_aggregated.append((dense_grad_indices[i], g))
  for i, g in enumerate(sparse_aggregated):
    indexed_aggregated.append((sparse_grad_indices[i], g))
  _, aggregated = zip(*sorted(indexed_aggregated))
  return aggregated


def _aggregate_dense(grads):
  r'''Aggregate dense gradients.
  '''
  return [Collective.get().allreduce(g) for g in grads]


def _aggregate_sparse(grads):
  r'''Aggregate sparse gradients.
  '''
  if not grads:
    return []
  aggregated_grads = []
  for g in grads:
    aggregated_values = Collective.get().allgather(g.values)
    aggregated_indices = Collective.get().allgather(g.indices)
    aggregated_grads.append(
      ops.IndexedSlices(
        aggregated_values,
        aggregated_indices,
        g.dense_shape))
  return aggregated_grads


def aggregate_gradients(grads_and_vars):
  r'''Aggregate gradients and variables.
  '''
  devices = Context.get().devices
  if len(devices) <= 1:
    return grads_and_vars

  replica_vars_to_optimize = []
  replica_grads = []
  replica_grad_indices = []
  sharded_vars_to_optimize = []
  shard_grads = []
  shard_grad_indices = []
  all_sharded_vars = ops.get_default_graph().get_collection(
    GraphKeys.SHARDED_VARIABLES)
  all_sharded_vars += ops.get_default_graph().get_collection(
    GraphKeys.NOT_REPLICATED)
  for i, gv in enumerate(grads_and_vars):
    if gv[0] is None:
      continue
    if gv[1] in all_sharded_vars:
      shard_grad_indices.append(i)
      shard_grads.append(gv[0])
      sharded_vars_to_optimize.append(gv[1])
    else:
      replica_grad_indices.append(i)
      replica_grads.append(gv[0])
      replica_vars_to_optimize.append(gv[1])
  replica_grads = _aggregate(
    replica_vars_to_optimize, replica_grads,
    sharded_vars_to_optimize, shard_grads)
  ops.get_collection_ref(GraphKeys.TRAINABLE_REPLICATED).extend(
    replica_vars_to_optimize)
  ops.get_collection_ref(GraphKeys.TRAINABLE_SHARDED).extend(
    sharded_vars_to_optimize)
  return _pack_grads_and_vars(
    _mean(replica_grads, devices), shard_grads,
    replica_vars_to_optimize, sharded_vars_to_optimize,
    replica_grad_indices, shard_grad_indices)

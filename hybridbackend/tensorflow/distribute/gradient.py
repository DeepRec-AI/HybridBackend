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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

from hybridbackend.tensorflow.distribute.communicator_pool import \
  CommunicatorPool
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys


def _tensor_size(tensor):
  r'''Calculate size of specific tensor.
  '''
  num_elements = tensor.shape.num_elements()
  if num_elements is not None and num_elements > 0:
    return tensor.dtype.size * num_elements
  return -1


def _log(tensors, sparse_tensors, shard_tensors):
  r'''Log values of gradients.
  Args:
    tensors: list of dense gradients.
    sparse_tensors: list of sparse gradients.
    shard_tensors: list of tensors skipped in aggregation.
  '''
  sizes = [_tensor_size(t) for t in tensors]
  nonempty_sizes = [s for s in sizes if s > 0]
  sparse_sizes = [
    _tensor_size(s.values) + _tensor_size(s.indices)
    for s in sparse_tensors]
  nonempty_sparse_sizes = [s for s in sparse_sizes if s > 0]
  grad_mbs = sum(nonempty_sizes) / 1024.0 / 1024.0
  sparse_grad_mbs = sum(nonempty_sparse_sizes) / 1024.0 / 1024.0
  logging.info(
    f'Aggregate {len(sizes)} dense gradients '
    f'({">" if len(sizes) > len(nonempty_sizes) else ""}{grad_mbs:.2f}MB) '
    f'and {len(sparse_sizes)} sparse gradients '
    f'({">" if len(sparse_sizes) > len(nonempty_sparse_sizes) else ""}'
    f'{sparse_grad_mbs:.2f}MB), '
    f'skip {len(shard_tensors)} aggregated gradients')


def _flatten(tensors):
  r'''Flatten list of tensors into one buffer tensor.
  Args:
    tensors: tensors to flatten.
  Returns:
    flattened: flattened buffer tensor.
    shapes: shapes of tensors.
    sizes: sizes of tensors.
  '''
  ftensors = [array_ops.reshape(t, [-1]) for t in tensors]
  shapes = [array_ops.shape(t) for t in tensors]
  sizes = [array_ops.reshape(array_ops.shape(t), []) for t in ftensors]
  return array_ops.concat(ftensors, 0), shapes, sizes


def _deflatten(flattened, shapes, sizes):
  r'''Get original tensor list from flattened buffer tensor.
  Args:
    flattened: flattened buffer tensor.
    shapes: shapes of tensors.
    sizes: sizes of tensors.
  Returns:
    original tensor list.
  '''
  ftensors = array_ops.split(flattened, sizes)
  return [array_ops.reshape(t, shapes[i]) for i, t in enumerate(ftensors)]


def _bucketize(grads, num_buckets):
  r'''Split gradients into buckets.
  Args:
    grads: gradients to split.
  Returns:
    list of tensor buckets.
  '''
  if not grads:
    return []
  if len(grads) == 1:
    return [grads]
  # Pass 1: split by dtypes.
  dtype_buckets = []
  prev_dtype = None
  for g in grads:
    if g.dtype != prev_dtype:
      dtype_buckets.append([g])
      prev_dtype = g.dtype
    else:
      dtype_buckets[-1].append(g)
  num_dtypes = len(dtype_buckets)
  if num_dtypes >= num_buckets:
    if num_dtypes > num_buckets:
      logging.warning(f'num_buckets is increased to {num_dtypes}')
    return dtype_buckets
  # Pass 2: split by sizes.
  # Pass 2.1: Calculate number of splits for different dtypes.
  dtype_bucket_sizes = [len(b) for b in dtype_buckets]
  dtype_bucket_total_size = sum(dtype_bucket_sizes)
  bucket_num_splits = [
    int(num_buckets * cnt / dtype_bucket_total_size)
    for cnt in dtype_bucket_sizes]
  bucket_num_splits = [n if n > 0 else 1 for n in bucket_num_splits]
  bucket_num_splits[0] += (num_buckets - sum(bucket_num_splits))
  buckets = []
  for bucket_id, bucket in enumerate(dtype_buckets):
    # Pass 2.2: For each dtype, calculate split size.
    bucket_info = [(g.dtype.size, g.shape.num_elements()) for g in bucket]
    bucket_sizes_or_none = [s * n if n else None for s, n in bucket_info]
    bucket_sizes_without_none = [s for s in bucket_sizes_or_none if s]
    bucket_min_total_size = sum(bucket_sizes_without_none)
    bucket_mean_size = bucket_min_total_size / len(bucket_sizes_without_none)
    bucket_sizes = [
      s if s else bucket_mean_size for s in bucket_sizes_or_none]
    if bucket_num_splits[bucket_id] == 1:
      split_size = sum(bucket_sizes)
    else:
      split_size = sum(bucket_sizes) / (bucket_num_splits[bucket_id] - 1)
    # Pass 2.3: For each dtype, split buckets by split size.
    ext_buckets = []
    prev_size = 0
    for gid, g in enumerate(bucket):
      if prev_size == 0 or prev_size + bucket_sizes[gid] > split_size:
        ext_buckets.append([g])
        prev_size = bucket_sizes[gid]
      else:
        ext_buckets[-1].append(g)
        prev_size += bucket_sizes[gid]
    buckets.extend(ext_buckets)
  buckets = [g for g in buckets if g]
  return buckets


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


def _aggregate(replica_grads, shard_grads, num_buckets):
  r'''Aggregate gradients.
  '''
  if not replica_grads:
    return []

  dense_grads = []
  dense_grad_indices = []
  sparse_grads = []
  sparse_grad_indices = []
  for idx, g in enumerate(replica_grads):
    if isinstance(g, ops.IndexedSlices):
      sparse_grads.append(g)
      sparse_grad_indices.append(idx)
    else:
      dense_grads.append(ops.convert_to_tensor(g))
      dense_grad_indices.append(idx)
  _log(dense_grads, sparse_grads, shard_grads)
  dense_aggregated = _aggregate_dense(dense_grads, num_buckets)
  sparse_aggregated = _aggregate_sparse(sparse_grads)
  indexed_aggregated = []
  for i, g in enumerate(dense_aggregated):
    indexed_aggregated.append((dense_grad_indices[i], g))
  for i, g in enumerate(sparse_aggregated):
    indexed_aggregated.append((sparse_grad_indices[i], g))
  _, aggregated = zip(*sorted(indexed_aggregated))
  return aggregated


def _aggregate_dense(grads, num_buckets):
  r'''Aggregate dense gradients.
  '''
  if not grads:
    return []
  if len(grads) == 1:
    aggregated = CommunicatorPool.get().allreduce(grads[0], trainable=False)
    return [aggregated]
  # Bucketize gradients.
  buckets = _bucketize(grads, num_buckets)
  # Flatten buckets.
  bucket_tensor_info = []
  for bucket_id, bucket in enumerate(buckets):
    with ops.name_scope(f'buckets/{bucket_id}'):
      bucket_tensor_info.append(_flatten(bucket))
  bucket_tensors, bucket_tensor_shapes, bucket_tensor_sizes = zip(
    *bucket_tensor_info)
  # Aggregate on flattened tensors.
  bucket_aggregations = []
  for bucket_id, flattened in enumerate(bucket_tensors):
    aggregated = CommunicatorPool.get().allreduce(flattened, trainable=False)
    bucket_aggregations.append(aggregated)
  # Unflatten buckets.
  bucket_grads = [
    _deflatten(
      g,
      bucket_tensor_shapes[bucket_id],
      bucket_tensor_sizes[bucket_id])
    for bucket_id, g in enumerate(bucket_aggregations)]
  # Unbucketize gradients.
  aggregated_grads = [g for bucket in bucket_grads for g in bucket]
  return aggregated_grads


def _aggregate_sparse(grads):
  r'''Aggregate sparse gradients.
  '''
  if not grads:
    return []
  aggregated_grads = []
  for g in grads:
    aggregated_values = CommunicatorPool.get().allgatherv(
      g.values, trainable=False)
    aggregated_indices = CommunicatorPool.get().allgatherv(
      g.indices, trainable=False)
    aggregated_grads.append(
      ops.IndexedSlices(
        aggregated_values,
        aggregated_indices,
        g.dense_shape))
  return aggregated_grads


def aggregate_gradients(grads_and_vars, num_buckets=1):
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
  replica_grads = _aggregate(replica_grads, shard_grads, num_buckets)
  ops.get_collection_ref(GraphKeys.TRAINABLE_REPLICATED).extend(
    replica_vars_to_optimize)
  ops.get_collection_ref(GraphKeys.TRAINABLE_SHARDED).extend(
    sharded_vars_to_optimize)
  return _pack_grads_and_vars(
    _mean(replica_grads, devices), _mean(shard_grads, devices),
    replica_vars_to_optimize, sharded_vars_to_optimize,
    replica_grad_indices, shard_grad_indices)

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

from hybridbackend.tensorflow.distribute.communicator import CollectiveOps
from hybridbackend.tensorflow.distribute.communicator_lib import \
  CommunicatorPool


class GradientAggregation(object):  # pylint: disable=useless-object-inheritance
  r'''Functor for gradient aggregation.
  '''
  def __init__(self, devices, num_buckets=1, name=None):
    self._devices = devices
    self._num_buckets = num_buckets
    if not name:
      name = 'aggregate_grads'
    self._name = name

  def _log(self, tensors, message):
    r'''Log values of tensors.

    Args:
      tensors: tensor list.
    '''
    dtypes = [t.dtype for t in tensors]
    sizes = [
      t.dtype.size * t.shape.num_elements()
      if t.shape.num_elements() is not None and t.shape.num_elements() > 0
      else None
      for t in tensors]
    nonempty_sizes = [s for s in sizes if s]
    logging.info(
      f'{message}: {len(tensors)} tensors '
      f'({", ".join([repr(dt) for dt in set(dtypes)])}): '
      f'{sum(nonempty_sizes) / 1024.0 / 1024.0:.2f} MB and '
      f'{len(sizes) - len(nonempty_sizes)} dynamic-shaped tensors')

  def _flatten(self, tensors):
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

  def _deflatten(self, flattened, shapes, sizes):
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

  def _bucketize(self, grads):
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
    if num_dtypes >= self._num_buckets:
      if num_dtypes > self._num_buckets:
        logging.warning(f'num_buckets is increased to {num_dtypes}')
      if len(dtype_buckets) > 1:
        for bucket_id, bucket in enumerate(dtype_buckets):
          self._log(
            bucket,
            (f'Aggregate {bucket_id + 1}/{len(dtype_buckets)} '
             'of gradient buckets'))
      return dtype_buckets

    # Pass 2: split by sizes.
    # Pass 2.1: Calculate number of splits for different dtypes.
    dtype_bucket_sizes = [len(b) for b in dtype_buckets]
    dtype_bucket_total_size = sum(dtype_bucket_sizes)
    bucket_num_splits = [
      int(self._num_buckets * cnt / dtype_bucket_total_size)
      for cnt in dtype_bucket_sizes]
    bucket_num_splits = [n if n > 0 else 1 for n in bucket_num_splits]
    bucket_num_splits[0] += (self._num_buckets - sum(bucket_num_splits))

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
    if len(buckets) > 1:
      for bucket_id, bucket in enumerate(buckets):
        self._log(
          bucket,
          f'Aggregate {bucket_id + 1}/{len(buckets)} of gradient buckets')

    return buckets

  def _allreduce(self, comm, inputs, inputs_deps):
    r'''Allreduce inputs.
    '''
    with ops.control_dependencies(inputs_deps):
      return comm.allreduce(inputs[0], reduce_op=CollectiveOps.SUM), None

  def _reduce(self, grads):
    r'''Reduce gradients.
    '''
    if grads is None or (isinstance(grads, (list, tuple)) and not grads):
      return []

    if (len(self._devices)) <= 1:
      return grads

    multiplier = 1. / len(self._devices)
    if grads is None:
      return None
    if isinstance(grads, list):
      return [self._reduce(g) for g in grads]
    if isinstance(grads, tuple):
      return tuple(self._reduce(g) for g in grads)
    if isinstance(grads, ops.Tensor):
      multiplier = math_ops.cast(multiplier, grads.dtype.base_dtype)
      return grads * multiplier
    if isinstance(grads, ops.IndexedSlices):
      return ops.IndexedSlices(
        self._reduce(grads.values),
        grads.indices,
        grads.dense_shape)
    raise ValueError(f'Type of {grads} is not supported')

  def _pack_grads_and_vars(
      self, replica_grads, shard_grads,
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

  def __call__(
      self, replica_grads, shard_grads,
      replica_vars, shard_vars,
      replica_grad_indices, shard_grad_indices):
    r'''Aggregate gradients.

    Args:
      replica_grads: Gradients replicated on devices.
      shard_grads: Gradients sharded on devices.

    Returns:
      aggregated_replica_grads: Aggregated gradients replicated on devices.
      aggregated_shard_grads: Aggregated gradients sharded on devices.
    '''
    if not replica_grads:
      return self._pack_grads_and_vars(
        [], self._reduce(shard_grads),
        replica_vars, shard_vars,
        replica_grad_indices, shard_grad_indices)

    replica_grads = [ops.convert_to_tensor(v) for v in replica_grads]
    self._log(replica_grads, 'Aggregate gradients')

    if len(replica_grads) == 1:
      aggregated = CommunicatorPool.get().call(
        self._allreduce, replica_grads[0], trainable=False)
      return self._pack_grads_and_vars(
        self._reduce([aggregated]), self._reduce(shard_grads),
        replica_vars, shard_vars, replica_grad_indices, shard_grad_indices)

    # Bucketize gradients.
    buckets = self._bucketize(replica_grads)

    # Flatten buckets.
    bucket_tensor_info = []
    for bucket_id, bucket in enumerate(buckets):
      with ops.name_scope(f'buckets/{bucket_id}'):
        bucket_tensor_info.append(self._flatten(bucket))
    bucket_tensors, bucket_tensor_shapes, bucket_tensor_sizes = zip(
      *bucket_tensor_info)

    # Aggregate on flattened tensors.
    bucket_aggregations = []
    for bucket_id, flattened in enumerate(bucket_tensors):
      aggregated = CommunicatorPool.get().call(
        self._allreduce, flattened, trainable=False)
      bucket_aggregations.append(aggregated)

    # Unflatten buckets.
    bucket_grads = [
      self._deflatten(
        g,
        bucket_tensor_shapes[bucket_id],
        bucket_tensor_sizes[bucket_id])
      for bucket_id, g in enumerate(bucket_aggregations)]

    # Unbucketize gradients.
    aggregated_grads = [g for bucket in bucket_grads for g in bucket]
    return self._pack_grads_and_vars(
      self._reduce(aggregated_grads), self._reduce(shard_grads),
      replica_vars, shard_vars, replica_grad_indices, shard_grad_indices)

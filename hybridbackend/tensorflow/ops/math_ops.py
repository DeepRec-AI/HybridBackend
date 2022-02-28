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

r'''Arithmetic operators for dense and sparse tensors.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops


def _get_segment_ids_and_num(ids, segment_rank):
  r'''Calculate segment ids and num_segments with a specifed segment_rank.
  '''
  if segment_rank < 0 or segment_rank > (ids.dense_shape.shape[0].value - 1):
    raise ValueError(
        f'invalid segment_rank {segment_rank}, must be within' + \
        f' [0, {ids.dense_shape.shape[0].value})')
  segment_ids = ids.indices[:, segment_rank]
  stride = ids.dense_shape[segment_rank]
  for i in reversed(xrange(segment_rank)):
    segment_ids = math_ops.add_n(
        [segment_ids, math_ops.multiply(ids.indices[:, i], stride)])
    stride = stride * ids.dense_shape[i]
  if segment_ids.dtype != dtypes.int32:
    segment_ids = math_ops.cast(segment_ids, dtypes.int32)
  return segment_ids, stride


def _segment_sum(ids, data, weights=None, pad=True, segment_rank=0, name=None):
  r'''Do segment sum with weights.

  Example:
    data: [1., 2., 3.]
    weights: [2., 2., 2.]
    ids.indices: [0, 1], [1, 2], [1, 3]
    ids.dense_shape: [4, 4]
    pad: True
    segment_rank: 0
    output: [2, 10, 0, 0]

  Args:
    ids: sparse_tensor.SparseTensor, has the same length as data.
    data: embedding values.
    weights: has the same length as data.
    pad: whether or not padding the output.
    segment_rank: segment reduce by values on this rank, default is 0.

  Returns:
    summation of data and weights by segments.
  '''
  if weights is not None:
    data = data * weights
  if pad:
    segment_ids, num_segments = _get_segment_ids_and_num(ids, segment_rank)
    return math_ops.unsorted_segment_sum(
        data, segment_ids, num_segments, name=name)
  segment_ids = ids.indices[:, 0]
  if segment_ids.dtype != dtypes.int32:
    segment_ids = math_ops.cast(segment_ids, dtypes.int32)
  return math_ops.segment_sum(data, segment_ids, name=name)


def _sparse_segment_sum(
    ids, data, indices, pad=True, segment_rank=0, name=None):
  r'''Segment sum with entries of data specifed by indices.

  Example:
    data: [1., 2., 3.]
    indices: [0, 1]
    ids.indices: [0, 1], [1, 2], [1, 3]
    ids.dense_shape: [4, 4]
    pad: True
    segment_rank: 0
    output: [1, 2, 0, 0]

  Args:
    ids: sparse_tensor.SparseTensor, has the same length as data.
    data: embedding values.
    indices: specify the entries in data to be summed up.
    pad: whether or not padding the output.
    segment_rank: segment reduce by values on this rank, default is 0

  Returns:
    summation of data by segments.
  '''
  if pad:
    segment_ids, num_segments = _get_segment_ids_and_num(ids, segment_rank)
    return math_ops.sparse_segment_sum_with_num_segments(
        data, indices, segment_ids, name=name, num_segments=num_segments)
  segment_ids = ids.indices[:, 0]
  if segment_ids.dtype != dtypes.int32:
    segment_ids = math_ops.cast(segment_ids, dtypes.int32)
  return math_ops.sparse_segment_sum(
      data, indices, segment_ids, name=name)


def _segment_mean(ids, data, weights=None, pad=True, segment_rank=0, name=None):
  r'''Segment mean with weights.

  Example:
    data: [1., 2., 3.]
    weights: [2., 2., 2.]
    ids.indices: [0, 1], [1, 2], [1, 3]
    ids.dense_shape: [4, 4]
    pad: True
    segment_rank: 0
    output: [1, 2.5, 0, 0]

  Args:
    ids: sparse_tensor.SparseTensor, has the same length as data.
    data: embedding values.
    weights: has the same length as data.
    pad: whether or not padding the output.
    segment_rank: segment reduce by values on this rank, default is 0

  Returns:
    mean of segment summed weighted data by segment summed weights.
  '''
  if weights is not None:
    data = data * weights
  if pad:
    segment_ids, num_segments = _get_segment_ids_and_num(ids, segment_rank)
    data_sum = math_ops.unsorted_segment_sum(
        data, segment_ids, num_segments)
    if weights is None:
      return data_sum
    weight_sum = math_ops.unsorted_segment_sum(
        weights, segment_ids, num_segments)
    return math_ops.div(data_sum, weight_sum, name=name)
  segment_ids = ids.indices[:, 0]
  if segment_ids.dtype != dtypes.int32:
    segment_ids = math_ops.cast(segment_ids, dtypes.int32)
  data_sum = math_ops.segment_sum(data, segment_ids)
  if weights is None:
    return data_sum
  weight_sum = math_ops.segment_sum(weights, segment_ids)
  return math_ops.div(data_sum, weight_sum, name=name)


def _sparse_segment_mean(
    ids, data, indices, pad=True, segment_rank=0, name=None):
  r'''Segment mean with entries of data specifed by indices.

  Example:
    data: [1., 2., 3.]
    indices: [0, 1]
    ids.indices: [0, 1], [1, 2], [1, 3]
    ids.dense_shape: [4, 4]
    pad: True
    segment_rank: 0
    output: [0.25, 0.5, 0, 0]

  Args:
    ids: sparse_tensor.SparseTensor, has the same length as data.
    data: embedding values.
    indices: specify the entries in data to be summed up.
    pad: whether or not padding the output.
    segment_rank: segment reduce by values on this rank, default is 0

  Returns:
    mean of segment summed data by segments.
  '''
  if pad:
    segment_ids, num_segments = _get_segment_ids_and_num(ids, segment_rank)
    return math_ops.sparse_segment_mean_with_num_segments(
        data, indices, segment_ids, name=name, num_segments=num_segments)
  segment_ids = ids.indices[:, 0]
  if segment_ids.dtype != dtypes.int32:
    segment_ids = math_ops.cast(segment_ids, dtypes.int32)
  return math_ops.sparse_segment_mean(
      data, indices, segment_ids, name=name)


def _segment_sqrtn(
    ids, data, weights=None, pad=True, segment_rank=0, name=None):
  r'''Segment sqrtn with weights.

  Example:
    data: [1., 2., 3.]
    weights: [2., 2., 2.]
    ids.indices: [0, 1], [1, 2], [1, 3]
    ids.dense_shape: [4, 4]
    pad: True
    segment_rank: 0
    output: [1, 3.57, 0, 0]

  Args:
    ids: sparse_tensor.SparseTensor, has the same length as data.
    data: embedding values.
    weights: has the same length as data.
    pad: whether or not padding the output.
    segment_rank: segment reduce by values on this rank, default is 0

  Returns:
    sqrtn of segment summed weighted data by segment summed weights.
  '''
  if weights is not None:
    data = data * weights
  if pad:
    segment_ids, num_segments = _get_segment_ids_and_num(ids, segment_rank)
    data_sum = math_ops.unsorted_segment_sum(
        data, segment_ids, num_segments)
    if weights is None:
      return data_sum
    weights_squared = math_ops.pow(weights, 2)
    weight_sum = math_ops.unsorted_segment_sum(
        weights_squared, segment_ids, num_segments)
    weight_sum_sqrt = math_ops.sqrt(weight_sum)
    return math_ops.div(data_sum, weight_sum_sqrt, name=name)
  segment_ids = ids.indices[:, 0]
  if segment_ids.dtype != dtypes.int32:
    segment_ids = math_ops.cast(segment_ids, dtypes.int32)
  data_sum = math_ops.segment_sum(data, segment_ids)
  if weights is None:
    return data_sum
  weights_squared = math_ops.pow(weights, 2)
  weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
  weight_sum_sqrt = math_ops.sqrt(weight_sum)
  return math_ops.div(data_sum, weight_sum_sqrt, name=name)


def _sparse_segment_sqrt_n(
    ids, data, indices, pad=True, segment_rank=0, name=None):
  r'''Segment sqrtn with entries of data specifed by indices.

  Example:
    data: [1., 2., 3.]
    indices: [0, 1]
    ids.indices: [0, 1], [1, 2], [1, 3]
    ids.dense_shape: [4, 4]
    pad: True
    segment_rank: 0
    output: [0.5, 1, 0, 0]

  Args:
    ids: sparse_tensor.SparseTensor, has the same length as data.
    data: embedding values.
    indices: specify the entries in data to be summed up.
    pad: whether or not padding the output.
    segment_rank: segment reduce by values on this rank, default is 0

  Returns:
    sqrtn of segment summed data by segments.
  '''
  if pad:
    segment_ids, num_segments = _get_segment_ids_and_num(ids, segment_rank)
    return math_ops.sparse_segment_sqrt_n_with_num_segments(
        data, indices, segment_ids, name=name, num_segments=num_segments)
  segment_ids = ids.indices[:, 0]
  if segment_ids.dtype != dtypes.int32:
    segment_ids = math_ops.cast(segment_ids, dtypes.int32)
  return math_ops.sparse_segment_sqrt_n(
      data, indices, segment_ids, name=name)


def _segment_tile(ids, data, weights=None, pad=False, name=None):
  r'''Segment tiling with weights.

  Example:
    data: [1., 2., 3.]
    weights: [2., 2., 2.]
    ids.indices: [0, 1], [1, 2], [1, 3]
    ids.dense_shape: [4, 2]
    pad: True
    num_tiles = 2
    tile_ids = [1, 2, 3]
    tiled_segment_ids = [1, 4, 5]
    segments = 8
    weighted_data = [2, 4, 6]
    output = [0, 2, 0, 0, 4, 6, 0, 0]

  Args:
    ids: sparse_tensor.SparseTensor, has the same length as data.
    data: embedding values.
    weights: has the same length as data.
    pad: whether or not padding the output.

  Returns:
    tiling of segment summed weighted data.
  '''
  dim = data.shape[1]
  if weights is not None:
    data = data * weights
  segment_ids = ids.indices[:, 0]
  if segment_ids.dtype != dtypes.int32:
    segment_ids = math_ops.cast(segment_ids, dtypes.int32)
  tile_ids = ids.indices[:, 1]
  if tile_ids.dtype != dtypes.int32:
    tile_ids = math_ops.cast(tile_ids, dtypes.int32)
  num_segments = ids.dense_shape[0]
  if num_segments.dtype != dtypes.int32:
    num_segments = math_ops.cast(num_segments, dtypes.int32)
  num_tiles = ids.dense_shape[1]
  if num_tiles.dtype != dtypes.int32:
    num_tiles = math_ops.cast(num_tiles, dtypes.int32)
  tiled_segment_ids = segment_ids * num_tiles + tile_ids
  if pad:
    segmented = math_ops.unsorted_segment_sum(
        data, tiled_segment_ids, num_segments * num_tiles, name=name)
  else:
    segmented = math_ops.segment_sum(
        data, tiled_segment_ids, name=name)
  return array_ops.reshape(segmented, [-1, num_tiles * dim])


def _sparse_segment_tile(ids, data, indices, pad=False, name=None):
  r'''Segment tiling with specified indices.

  Example:
    data: [1., 2., 3.]
    indices: [0, 1]
    ids.indices: [0, 1], [1, 2], [1, 3]
    ids.dense_shape: [4, 2]
    pad: True
    num_tiles = 2
    tile_ids = [1, 2, 3]
    tiled_segment_ids = [1, 4, 5]
    segments = 8
    output = [0, 1, 0, 0, 4, 0, 0, 0]

  Args:
    ids: sparse_tensor.SparseTensor, has the same length as data.
    data: embedding values.
    indices: specify the entries in data to be summed up.
    pad: whether or not padding the output.

  Returns:
    tiling of segment summed data by segments.
  '''
  segment_ids = ids.indices[:, 0]
  if segment_ids.dtype != dtypes.int32:
    segment_ids = math_ops.cast(segment_ids, dtypes.int32)
  tile_ids = ids.indices[:, 1]
  num_tiles = ids.dense_shape[1]
  tiled_segment_ids = segment_ids * num_tiles + tile_ids
  if pad:
    num_segments = ids.dense_shape[0]
    return math_ops.sparse_segment_sum_with_num_segments(
        data, indices, tiled_segment_ids,
        name=name,
        num_segments=num_segments * num_tiles)
  return math_ops.sparse_segment_sum(
      data, indices, tiled_segment_ids, name=name)


def _segment_combine(
    ids, data,
    weights=None,
    pad=True,
    segment_rank=0,
    combiner=None,
    name=None):
  r'''Computes the sum along segments of a tensor.

  Args:
    ids: A `SparseTensor` as original input.
    data: A `Tensor` with data that will be assembled in the output.
    weights: Weights for data.
    pad: If True, pad the output.
    segment_rank: segment reduce by values on this rank, default is 0
    combiner: A string specifying the reduction op.
    name: Optional name of the operation.
  '''
  combiner = combiner or "sum"
  with ops.name_scope(combiner):
    if combiner == "sum":
      return _segment_sum(
          ids, data, weights, pad=pad, segment_rank=segment_rank, name=name)
    if combiner == "mean":
      return _segment_mean(
          ids, data, weights, pad=pad, segment_rank=segment_rank, name=name)
    if combiner == "sqrtn":
      return _segment_sqrtn(
          ids, data, weights, pad=pad, segment_rank=segment_rank, name=name)
    if combiner == "tile":
      return _segment_tile(ids, data, weights, pad=pad, name=name)
    if callable(combiner):
      return combiner(
          ids, data,
          weights=weights,
          indices=None,
          pad=pad,
          name=name)
  raise ValueError(f'Unrecognized combiner: {combiner}')


def _sparse_segment_combine(
    ids, data, indices,
    pad=True,
    segment_rank=0,
    combiner=None,
    name=None):
  r'''Computes the sum along sparse segments of a tensor.

  Args:
    ids: A `SparseTensor` as original input.
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    pad: If True, pad the output.
    segment_rank: segment reduce by values on this rank, default is 0
    combiner: A string specifying the reduction op.
    name: Optional name of the operation.
  '''
  combiner = combiner or "sum"
  with ops.name_scope(combiner):
    if combiner == "sum":
      return _sparse_segment_sum(
          ids, data, indices,
          pad=pad,
          segment_rank=segment_rank,
          name=name)
    if combiner == "mean":
      return _sparse_segment_mean(
          ids, data, indices,
          pad=pad,
          segment_rank=segment_rank,
          name=name)
    if combiner == "sqrtn":
      return _sparse_segment_sqrt_n(
          ids, data, indices,
          pad=pad,
          segment_rank=segment_rank,
          name=name)
    if combiner == "tile":
      return _sparse_segment_tile(
          ids, data, indices,
          pad=pad,
          name=name)
    if callable(combiner):
      return combiner(
          ids, data,
          weights=None,
          indices=indices,
          pad=pad,
          name=name)
    raise ValueError(f'Unrecognized combiner: {combiner}')


def segment_combine(
    ids, data,
    weights=None,
    indices=None,
    dimension=None,
    pad=True,
    segment_rank=0,
    combiner=None,
    name=None):
  r'''Computes the sum along segments of a tensor.

  Args:
    ids: A `SparseTensor` as original input.
    data: A `Tensor` with data that will be assembled in the output.
    weights: Weights with same shape of data.
    indices: A 1-D `Tensor` with indices into `data`.
    dimension: Dimension of embedding weights.
    pad: If True, pad the output.
    segment_rank: segment reduce by values on this rank, default is 0.
    combiner: A string specifying the reduction op.
    name: Optional name of the operation.
  '''
  if name is None:
    name = 'segment_combine'

  if data.dtype in (dtypes.float16, dtypes.bfloat16):
    data = math_ops.cast(data, dtypes.float32)

  if not isinstance(ids, sparse_tensor.SparseTensor):
    # Reshape to reverse the flattening of ids.
    with ops.name_scope('update_shape'):
      dims = tensor_shape.TensorShape([dimension])
      embeddings_shape = ids.get_shape().concatenate(dims)
      ids_shape = array_ops.shape(ids, name='ids_shape')
      if dims.is_fully_defined():
        actual_embedding_shape = array_ops.concat(
            [ids_shape, dims], 0, name='actual_embedding_shape')
      else:
        actual_embedding_shape = array_ops.concat(
            [ids_shape, array_ops.shape(data)[1:]], 0,
            name='actual_embedding_shape')
    ## recover unique
    if indices is not None:
      data = array_ops.gather(data, indices)
    embeddings = array_ops.reshape(
        data, actual_embedding_shape, name=name)
    embeddings.set_shape(embeddings_shape)
    return embeddings

  if weights is not None:
    if indices is not None:
      data = array_ops.gather(data, indices)
    weights = weights.values

    if weights.dtype != data.dtype:
      weights = math_ops.cast(weights, data.dtype)

    # Reshape weights to allow broadcast
    ones = array_ops.fill(
        array_ops.expand_dims(array_ops.rank(data) - 1, 0), 1)
    bcast_weights_shape = array_ops.concat(
        [array_ops.shape(weights), ones], 0)
    orig_weights_shape = weights.get_shape()
    weights = array_ops.reshape(weights, bcast_weights_shape)

    # Set the weight shape, since after reshaping to
    # bcast_weights_shape, the shape becomes None.
    if data.get_shape().ndims is not None:
      weights.set_shape(
          orig_weights_shape.concatenate(
              [1 for _ in xrange(data.get_shape().ndims - 1)]))

    return _segment_combine(
        ids, data,
        weights=weights,
        pad=pad,
        segment_rank=segment_rank,
        combiner=combiner,
        name=name)
  if indices is not None:
    return _sparse_segment_combine(
        ids, data, indices,
        pad=pad,
        segment_rank=segment_rank,
        combiner=combiner,
        name=name)
  return _segment_combine(
      ids, data,
      weights=None,
      pad=pad,
      segment_rank=segment_rank,
      combiner=combiner,
      name=name)


def floormod_partition(ids, num_shards):
  r'''Partition IDs.

  Args:
    ids: Input tensor to partition.
    num_shards: Number of shards to partition.
  '''
  ids_shards_indices = math_ops.floormod(
      math_ops.cast(ids, dtypes.int32),
      num_shards)
  partitioned_ids = data_flow_ops.dynamic_partition(
      ids // num_shards,
      ids_shards_indices,
      num_shards)
  ids_indices = math_ops.range(array_ops.size(ids))
  partitioned_indices = data_flow_ops.dynamic_partition(
      ids_indices, ids_shards_indices, num_shards)
  partitioned_ids = tuple(
      array_ops.reshape(v, [-1]) for v in partitioned_ids)
  partitioned_indices = tuple(
      array_ops.reshape(v, [-1]) for v in partitioned_indices)
  return partitioned_ids, partitioned_indices

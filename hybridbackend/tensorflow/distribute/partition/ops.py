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

r'''Partitioning message for collective ops.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops

from hybridbackend.tensorflow.framework.view import OperationLike


def partition(ids, num_partitions):
  r'''Partition IDs using floormod strategy.

   Args:
     ids: Input tensor to partition.
     num_partitions: Number of partitions.

  Return:
    output: Partitioned tensors.
    indices: Indices for stitching back.
  '''
  ids_shards_indices = math_ops.floormod(
    math_ops.cast(ids, dtypes.int32), num_partitions)
  partitioned_ids = data_flow_ops.dynamic_partition(
    ids, ids_shards_indices, num_partitions)
  ids_indices = math_ops.range(array_ops.size(ids))
  partitioned_indices = data_flow_ops.dynamic_partition(
    ids_indices, ids_shards_indices, num_partitions)
  partitioned_ids = [array_ops.reshape(v, [-1]) for v in partitioned_ids]
  partitioned_indices = [
    array_ops.reshape(v, [-1]) for v in partitioned_indices]
  return partitioned_ids, partitioned_indices


def _partition_by_modulo(ids, num_partitions, name=None):
  r'''Shuffle IDs using floormod strategy.

  Args:
    ids: Input tensor to partition.
    num_partitions: Number of partitions.
    name: Name of the operator.

  Return:
    output: A tensor with shuffled IDs.
    sizes: Size of each shard in output.
    indices: Indices for gathering back.
  '''
  if name is None:
    name = 'partition_by_modulo'

  with ops.device(ids.device):
    with ops.name_scope(name):
      return (
        OperationLike('PartitionByModulo')
        .returns_tensors(
          tensor_spec.TensorSpec(shape=ids.shape, dtype=ids.dtype),
          tensor_spec.TensorSpec(shape=[num_partitions], dtype=dtypes.int32),
          tensor_spec.TensorSpec(shape=ids.shape, dtype=dtypes.int32))
        .finalize(
          ids, num_partitions=num_partitions,
          name=name))


def partition_by_modulo(ids, num_partitions, name=None):
  r'''Shuffle IDs using floormod strategy.

  Args:
    ids: Input tensor to partition.
    num_partitions: Number of partitions.
    name: Name of the operator.

  Return:
    output: A tensor with shuffled IDs.
    sizes: Size of each shard in output.
    indices: Indices for gathering back.
  '''
  if name is None:
    name = 'partition_by_modulo'

  return _partition_by_modulo(
    ids, num_partitions=num_partitions, name=name)

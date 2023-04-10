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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec

from hybridbackend.tensorflow.framework.view import OperationLike


def _partition_by_dual_modulo_stage_one(
    ids, num_partitions, modulus, name=None):
  r'''Shuffle IDs using a two-staged
    (local modulo and global modulo) strategy.

  Args:
    ids: Input tensor to partition.
    num_partitions: Number of partitions.
    modulus: Size of a partition.
    name: Name of the operator.


  Return:
    output: A tensor with shuffled IDs.
    sizes: Size of each shard in output.
    indices: Indices for gathering back.
  '''
  if name is None:
    name = 'partition_by_dual_modulo_stage_one'

  with ops.device(ids.device):
    with ops.name_scope(name):
      return (
        OperationLike('PartitionByDualModuloStageOne')
        .returns_tensors(
          tensor_spec.TensorSpec(shape=ids.shape, dtype=ids.dtype),
          tensor_spec.TensorSpec(shape=[num_partitions], dtype=dtypes.int32),
          tensor_spec.TensorSpec(shape=ids.shape, dtype=dtypes.int32))
        .finalize(
          ids, num_partitions=num_partitions,
          modulus=modulus,
          name=name))


def partition_by_dual_modulo_stage_one(
    ids, num_partitions, modulus, name=None):
  r'''Shuffle IDs using a two-staged
    (local modulo and global modulo) strategy.

  Args:
    ids: Input tensor to partition.
    num_partitions: Number of partitions.
    modulus: Size of a partition.
    name: Name of the operator.


  Return:
    output: A tensor with shuffled IDs.
    sizes: Size of each shard in output.
    indices: Indices for gathering back.
  '''
  if name is None:
    name = 'partition_by_dual_modulo_stage_one'

  return _partition_by_dual_modulo_stage_one(
    ids, num_partitions=num_partitions,
    modulus=modulus, name=name)


def _partition_by_dual_modulo_stage_two(
    ids, num_partitions, modulus, name=None):
  r'''Shuffle IDs using a two-staged
    (local modulo and global modulo) strategy.

  Args:
    ids: Input tensor to partition.
    num_partitions: Number of partitions.
    modulus: Size of a partition.
    name: Name of the operator.


  Return:
    output: A tensor with shuffled IDs.
    sizes: Size of each shard in output.
    indices: Indices for gathering back.
  '''
  if name is None:
    name = 'partition_by_dual_modulo_stage_two'

  with ops.device(ids.device):
    with ops.name_scope(name):
      return (
        OperationLike('PartitionByDualModuloStageTwo')
        .returns_tensors(
          tensor_spec.TensorSpec(shape=ids.shape, dtype=ids.dtype),
          tensor_spec.TensorSpec(shape=[num_partitions], dtype=dtypes.int32),
          tensor_spec.TensorSpec(shape=ids.shape, dtype=dtypes.int32))
        .finalize(
          ids, num_partitions=num_partitions,
          modulus=modulus,
          name=name))


def partition_by_dual_modulo_stage_two(
    ids, num_partitions, modulus, name=None):
  r'''Shuffle IDs using a two-staged
    (local modulo and global modulo) strategy.

  Args:
    ids: Input tensor to partition.
    num_partitions: Number of partitions.
    modulus: Size of a partition.
    name: Name of the operator.


  Return:
    output: A tensor with shuffled IDs.
    sizes: Size of each shard in output.
    indices: Indices for gathering back.
  '''
  if name is None:
    name = 'partition_by_dual_modulo_stage_two'

  return _partition_by_dual_modulo_stage_two(
    ids, num_partitions=num_partitions,
    modulus=modulus, name=name)

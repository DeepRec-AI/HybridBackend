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

r'''A dataset that syncs data between replicas.

This class is compatible with Tensorflow 1.12.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec

from hybridbackend.tensorflow.common import oplib as _ops
from hybridbackend.tensorflow.data.sync.hook import SyncReplicasDatasetHook
from hybridbackend.tensorflow.data.sync.utils import denormalize
from hybridbackend.tensorflow.data.sync.utils import normalize


class _SyncReplicasDatasetV1(dataset_ops.Dataset):
  r'''A dataset that syncs data between replicas.
  '''
  def __init__(self, input_dataset, output_kinds):
    r'''Create a `_SyncReplicasDatasetV1`.

    Args:
      input_dataset: A `dataset` to be wrapped to verify its last element.
      output_kinds: A list of int to specify the type of tensors.
    '''
    self._input_dataset = input_dataset
    self._output_kinds = output_kinds
    super().__init__()

  def _as_variant_tensor(self):
    return _ops.hb_sync_replicas_dataset(
      self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
      output_kinds=self._output_kinds)  # pylint: disable=protected-access

  def _inputs(self):
    return [self._input_dataset]

  @property
  def output_shapes(self):
    return tensor_shape.TensorShape([]), self._input_dataset.output_shapes

  @property
  def output_types(self):
    return dtypes.int32, self._input_dataset.output_types

  @property
  def output_classes(self):
    return ops.Tensor, self._input_dataset.output_classes


class SyncReplicasDatasetV1(dataset_ops.Dataset):
  r'''A dataset that syncs data between replicas.
  '''
  @classmethod
  def apply(cls, world, rank):
    r'''Sync replicas for input dataset.
    '''
    def _apply_fn(dataset):
      return cls(dataset, world, rank)
    return _apply_fn

  @classmethod
  def hooks(cls):
    return SyncReplicasDatasetHook.all_instances()

  @classmethod
  def sync(cls, features=None):
    return SyncReplicasDatasetHook.sync_all_instances(features)

  def __init__(self, input_dataset, world=None, rank=None):
    r'''Create a `SyncReplicasDatasetV1`.

    Args:
      input_dataset: A `dataset` to be wrapped to verify its last element.
    '''
    self._input_dataset = input_dataset
    self._hook = SyncReplicasDatasetHook(world, rank)\
      if (world is not None and rank is not None) else None

    normalized_dataset, normalized_kinds, kinds = normalize(input_dataset)
    sync_replicas_dataset = _SyncReplicasDatasetV1(
      normalized_dataset, normalized_kinds)
    self._impl = denormalize(
      sync_replicas_dataset, self.element_spec, kinds, hook=self._hook)
    super().__init__()

  def _as_variant_tensor(self):
    return self._impl._as_variant_tensor()  # pylint: disable=protected-access

  def _inputs(self):
    return [self._input_dataset]

  @property
  def element_spec(self):
    if self._hook is None:
      return tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32), \
        self._input_dataset.element_spec
    return self._input_dataset.element_spec

  @property
  def output_shapes(self):
    if self._hook is None:
      return tensor_shape.TensorShape([]), self._input_dataset.output_shapes
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    if self._hook is None:
      return dtypes.int32, self._input_dataset.output_types
    return self._input_dataset.output_types

  @property
  def output_classes(self):
    if self._hook is None:
      return ops.Tensor, self._input_dataset.output_classes
    return self._input_dataset.output_classes

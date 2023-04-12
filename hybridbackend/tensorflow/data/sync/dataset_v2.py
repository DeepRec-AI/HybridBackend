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

This class is compatible with Tensorflow 1.15.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec

from hybridbackend.tensorflow.common import oplib as _ops
from hybridbackend.tensorflow.data.sync.hook import SyncReplicasDatasetHook
from hybridbackend.tensorflow.data.sync.utils import denormalize
from hybridbackend.tensorflow.data.sync.utils import normalize


class _SyncReplicasDatasetV2(dataset_ops.DatasetV2):
  r'''A dataset that syncs data between replicas.
  '''
  def __init__(self, input_dataset, output_kinds):
    r'''Create a `_SyncReplicasDatasetV2`.

    Args:
      input_dataset: A `dataset` to be wrapped to verify its last element.
      output_kinds: A list of int to specify the type of tensors.
    '''
    self._input_dataset = input_dataset
    self._should_stop_spec = tensor_spec.TensorSpec(
      shape=[], dtype=dtypes.int32)
    self._output_kinds = output_kinds
    variant_tensor = _ops.hb_sync_replicas_dataset(
      self._input_dataset._variant_tensor,
      output_kinds=output_kinds)
    super().__init__(variant_tensor)

  def _inputs(self):
    return [self._input_dataset]

  @property
  def element_spec(self):
    return self._should_stop_spec, self._input_dataset.element_spec  # pylint: disable=protected-access


class SyncReplicasDatasetV2(dataset_ops.DatasetV2):
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
    r'''Create a `SyncReplicasDatasetV2`.

    Args:
      input_dataset: A `dataset` to be wrapped to verify its last element.
    '''
    self._input_dataset = input_dataset
    self._hook = SyncReplicasDatasetHook(world, rank)\
      if (world is not None and rank is not None) else None
    normalized_dataset, normalized_kinds, kinds = normalize(input_dataset)
    sync_replicas_dataset = _SyncReplicasDatasetV2(
      normalized_dataset, normalized_kinds)
    self._impl = denormalize(
      sync_replicas_dataset, self.element_spec, kinds, hook=self._hook)
    super().__init__(self._impl._variant_tensor)  # pylint: disable=protected-access

  def _inputs(self):
    return [self._input_dataset]

  @property
  def element_spec(self):
    if self._hook is None:
      return tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32), \
        self._input_dataset.element_spec  # pylint: disable=protected-access
    return self._input_dataset.element_spec

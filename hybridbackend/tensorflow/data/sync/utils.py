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

r'''SyncReplicasDataset that reports the existence of next element.

This class is compatible with Tensorflow 1.12.
'''

from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

from hybridbackend.tensorflow.framework.ops import TensorKinds


def normalize(input_dataset):
  r'''flattent to normalize tensors within the input_dataset.
  '''
  flattened_specs = nest.flatten(input_dataset.element_spec)
  flattened_kinds = []
  for spec in flattened_specs:
    if isinstance(spec, tensor_spec.TensorSpec):
      flattened_kinds.append(TensorKinds.VALUES)
    elif isinstance(spec, sparse_tensor.SparseTensorSpec):
      flattened_kinds.append(
        sparse_tensor.SparseTensorValue(
          TensorKinds.INDICES, TensorKinds.VALUES, TensorKinds.DENSE_SHAPE))
    else:
      raise ValueError(
        'SyncReplicasDataset cannot support input datasets with outputs '
        'other than tensors or sparse tensors')
  return input_dataset.map(TensorKinds.normalize),\
    nest.flatten(flattened_kinds), flattened_kinds


def denormalize(input_dataset, element_spec, kinds, hook=None):
  r'''denormalize all tensors returned by input_dataset.
  '''
  if hook is None:
    return input_dataset.map(
      lambda *args: TensorKinds.denormalize(
        element_spec, [TensorKinds.VALUES] + kinds, args))
  input_dataset = input_dataset.map(hook.register)
  return input_dataset.map(
    lambda *args: TensorKinds.denormalize(
      element_spec, kinds, args))

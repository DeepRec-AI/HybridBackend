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

r'''DetectEndDataset that reports the existence of next element.

This class is compatible with Tensorflow 1.12.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from hybridbackend.tensorflow.common import oplib as _ops


class DetectEndDatasetV1(dataset_ops.Dataset):
  r'''Wrapping a dataset to notify whether it still has next input.
  '''
  def __init__(self, input_dataset):
    r'''Create a `_DetectEndDatasetV1`.

    Args:
      input_dataset: A `dataset` to be wrapped to verify its last element.
    '''
    self._input_dataset = input_dataset
    super().__init__()

  def _as_variant_tensor(self):
    return _ops.hb_detect_end_dataset(
      self._input_dataset._as_variant_tensor())  # pylint: disable=protected-access

  def _inputs(self):
    return [self._input_dataset]

  @property
  def output_shapes(self):
    return tensor_shape.TensorShape([]), self._input_dataset.output_shapes

  @property
  def output_types(self):
    return dtypes.bool, self._input_dataset.output_types

  @property
  def output_classes(self):
    return ops.Tensor, self._input_dataset.output_classes

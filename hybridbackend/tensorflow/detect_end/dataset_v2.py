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

This class is compatible with Tensorflow 1.15.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec

from hybridbackend.tensorflow.common import oplib as _ops


class DetectEndDatasetV2(dataset_ops.DatasetV2):
  r'''Wrapping a dataset to notify whether it still has next input.
  '''
  def __init__(self, input_dataset):
    r'''Create a `_DetectEndDatasetV2`.

    Args:
      input_dataset: A `dataset` to be wrapped to verify its last element.
    '''
    self._input_dataset = input_dataset
    self._marker_spec = tensor_spec.TensorSpec(shape=[], dtype=dtypes.bool)
    variant_tensor = _ops.hb_detect_end_dataset(
      self._input_dataset._variant_tensor)  # pylint: disable=protected-access
    super().__init__(variant_tensor)

  def _inputs(self):
    return [self._input_dataset]

  @property
  def element_spec(self):
    return self._marker_spec, self._input_dataset.element_spec  # pylint: disable=protected-access

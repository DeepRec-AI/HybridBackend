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

r'''Dataset that resizes batches of DataFrame values.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

# pylint: disable=ungrouped-imports
from hybridbackend.tensorflow.data.dataframe import input_fields

try:
  from tensorflow.python.data.ops.dataset_ops import DatasetV2 as _dataset  # pylint: disable=unused-import, line-too-long # noqa: F401

  from hybridbackend.tensorflow.data.rebatch.dataset_v2 import \
    RebatchDatasetV2 as RebatchDataset
  if inspect.isabstract(RebatchDataset):
    raise ImportError
  RebatchDataset.__module__ = __name__
  RebatchDataset.__name__ = 'RebatchDataset'
except ImportError:
  from hybridbackend.tensorflow.data.rebatch.dataset_v1 import \
    RebatchDatasetV1 as RebatchDataset
  RebatchDataset.__module__ = __name__
  RebatchDataset.__name__ = 'RebatchDataset'
  assert not inspect.isabstract(RebatchDataset)
# pylint: enable=ungrouped-imports


def rebatch(
    batch_size,
    drop_remainder=False,
    fields=None):
  r'''Create a `RebatchDataset`.

  Args:
    batch_size: Maxium number of samples in an output batch.
    drop_remainder: (Optional.) If True, smaller final batch is dropped.
      `False` by default.
    fields: (Optional.) List of DataFrame fields. Fetched from `input_dataset`
      by default.
  '''
  def _apply_fn(dataset):
    return RebatchDataset(
      dataset, input_fields(dataset, fields), batch_size,
      drop_remainder=drop_remainder)
  return _apply_fn

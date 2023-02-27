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

r'''Dataset that resizes batches of tensors.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

# pylint: disable=ungrouped-imports
try:
  from tensorflow.python.data.ops.dataset_ops import DatasetV2 as _dataset  # pylint: disable=unused-import

  from hybridbackend.tensorflow.data.rectify.dataset_v2 import \
    RectifyDatasetV2 as RectifyDataset
  if inspect.isabstract(RectifyDataset):
    raise ImportError
  RectifyDataset.__module__ = __name__
  RectifyDataset.__name__ = 'RectifyDataset'
except ImportError:
  from hybridbackend.tensorflow.data.rectify.dataset_v1 import \
    RectifyDatasetV1 as RectifyDataset
  RectifyDataset.__module__ = __name__
  RectifyDataset.__name__ = 'RectifyDataset'
  assert not inspect.isabstract(RectifyDataset)
# pylint: enable=ungrouped-imports


def rectify(batch_size, shuffle_buffer_size=None, drop_remainder=False):
  r'''Create a `RectifyDataset`.

  Args:
    batch_size: Maxium number of samples in an output batch.
    shuffle_buffer_size: (Optional.) A `tf.int64` scalar `tf.Tensor`,
      representing the number of elements from this dataset from which the new
      dataset will sample.
    drop_remainder: (Optional.) If True, smaller final batch is dropped.
      `False` by default.
  '''
  def _apply_fn(dataset):
    return RectifyDataset(
      dataset, batch_size,
      shuffle_buffer_size=shuffle_buffer_size,
      drop_remainder=drop_remainder)
  return _apply_fn

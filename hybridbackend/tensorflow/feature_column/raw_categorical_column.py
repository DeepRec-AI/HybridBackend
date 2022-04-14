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

r'''Feature column for raw categorical features.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import parsing_ops


class RawCategoricalColumn(fc.CategoricalColumn):
  r'''See `raw_categorical_column`.
  '''

  def __init__(self, key, dtype=dtypes.int64):
    r'''Constructor.

    Args:
      key: A unique string identifying the input feature. It is used as the
        column name and the dictionary key for feature parsing configs, feature
        `Tensor` objects, and feature columns.
      dtype: Data type of the column.
    '''
    self._key = key
    self._dtype = dtype

  @property
  def _is_v2_column(self):
    r'''See 'FeatureColumn` base class.
    '''
    return True

  @property
  def parents(self):
    r'''See 'FeatureColumn` base class.
    '''
    return [self._key]

  def _get_config(self):
    r'''See 'FeatureColumn` base class.
    '''
    return dict(zip(self._fields, self))

  @property
  def name(self):
    r'''See `FeatureColumn` base class.
    '''
    return self._key

  @property
  def num_buckets(self):
    r'''See `CategoricalColumn` base class.
    '''
    return None

  @property
  def _num_buckets(self):
    return self.num_buckets

  @property
  def parse_example_spec(self):
    r'''See `FeatureColumn` base class.
    '''
    return {self._key: parsing_ops.VarLenFeature(self._dtype)}

  def transform_feature(self, transformation_cache, state_manager):  # pylint: disable=unused-argument
    r'''Returns IDs with identity values.
    '''
    input_tensor = transformation_cache._features[self._key]  # pylint: disable=protected-access
    if not input_tensor.dtype.is_integer:
      raise ValueError(
        'Invalid input, not integer. key: '
        f'{self._key} dtype: {input_tensor.dtype}')
    return input_tensor

  def get_sparse_tensors(self, transformation_cache, state_manager):
    r'''See `CategoricalColumn` base class.
    '''
    return fc.CategoricalColumn.IdWeightPair(
      transformation_cache.get(self, state_manager), None)

  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    r'''See `CategoricalColumn` base class.
    '''
    del weight_collections
    del trainable
    return fc.CategoricalColumn.IdWeightPair(
      inputs.get(self._key), None)


def raw_categorical_column(key):
  r'''A `CategoricalColumn` that returns unchecked values.

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
  '''
  return RawCategoricalColumn(key)

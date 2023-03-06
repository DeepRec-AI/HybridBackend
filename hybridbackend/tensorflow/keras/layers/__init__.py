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

r'''HybridBackend Keras Layers.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  from tensorflow.python.feature_column.dense_features import DenseFeatures
except ImportError:
  pass


def dense_features(features, feature_columns):
  r'''Function produces dense tensors based on given `feature_columns`.

  Args:
    features: A mapping from key to tensors. `FeatureColumn`s look up via
      these keys. For example `numeric_column('price')` will look at 'price'
      key in this dict. Values can be a `SparseTensor` or a `Tensor` depends
      on corresponding `FeatureColumn`.
    feature_columns: List of feature columns.

  Returns:
    List of `Tensor`s which represents input layer of a model, which matches
    order of columns in `feature_columns`.
  '''
  cols_to_output_tensors = {}
  DenseFeatures(feature_columns)(
    features, cols_to_output_tensors=cols_to_output_tensors)
  return [cols_to_output_tensors[f] for f in feature_columns]

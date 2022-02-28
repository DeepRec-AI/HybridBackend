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

r'''A layer that produces a dense `Tensor` based on given `feature_columns`.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
try:
  from tensorflow.python.feature_column.dense_features import \
      DenseFeatures as _DenseFeatures
except ImportError:
  from tensorflow.python.feature_column.feature_column_v2 import \
      FeatureLayer as _DenseFeatures
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.ops import array_ops

from hybridbackend.tensorflow.feature_column.embedding_backend import \
    EmbeddingBackend
from hybridbackend.tensorflow.feature_column.embedding_backend_default import \
    EmbeddingBackendDefault  # pylint: disable=unused-import
from hybridbackend.tensorflow.feature_column.embedding_lookup_coalesced \
    import EmbeddingLookupCoalesced
from hybridbackend.tensorflow.feature_column.feature_column import \
    StateManagerImpl


class DenseFeatures(_DenseFeatures):
  r'''A layer that produces a dense `Tensor` based on given `feature_columns`.
  '''
  def __init__(self, feature_columns, trainable=True, name=None, **kwargs):
    r'''Constructs a DenseFeatures layer.
    '''
    self._impl = EmbeddingBackend.get()
    self._num_groups = kwargs.pop('num_groups', self._impl.num_groups())
    self._enable_concat = kwargs.pop(
        'enable_concat', self._impl.enable_concat())
    super().__init__(
        feature_columns=feature_columns,
        trainable=trainable,
        name=name,
        **kwargs)
    self._lookup_sparse_coalesced = EmbeddingLookupCoalesced()
    self._state_manager = StateManagerImpl(self, self.trainable)  # pylint: disable=protected-access

  @property
  def num_groups(self):
    r'''Number of embedding column groups.
    '''
    return self._num_groups

  def call(self, features, cols_to_output_tensors=None):
    r'''Returns a dense tensor corresponding to the `feature_columns`.

    Args:
      features: A mapping from key to tensors. `FeatureColumn`s look up via
        these keys. For example `numeric_column('price')` will look at 'price'
        key in this dict. Values can be a `SparseTensor` or a `Tensor` depends
        on corresponding `FeatureColumn`.
      cols_to_output_tensors: If not `None`, this will be filled with a dict
        mapping feature columns to output tensors created.

    Returns:
      A `Tensor` which represents input layer of a model. Its shape
      is (batch_size, first_layer_dimension) and its dtype is `float32`.
      first_layer_dimension is determined based on given `feature_columns`.
      If emb_enable_concat is disabled, `None` would be returned.

    Raises:
      ValueError: If arguments are not valid.
    '''
    if self.num_groups is None:
      return super().call(
          features, cols_to_output_tensors=cols_to_output_tensors)

    if not isinstance(features, dict):
      raise ValueError('We expected a dictionary here. Instead we got: ',
                       features)
    transformation_cache = fc.FeatureTransformationCache(features)
    indexed_coalesced_columns = []
    indexed_non_coalesced_columns = []
    for cid, c in enumerate(self._feature_columns):
      if self._impl.sharded(c):
        indexed_coalesced_columns.append(tuple([cid, c]))
      else:
        indexed_non_coalesced_columns.append(tuple([cid, c]))

    non_coalesced_column_output_tensors = []
    for cid, c in indexed_non_coalesced_columns:
      with ops.name_scope(c.name):
        tensor = c.get_dense_tensor(transformation_cache,
                                    self._state_manager)
        if hasattr(self, '_process_dense_tensor'):
          processed_tensors = self._process_dense_tensor(c, tensor)
        else:
          processed_tensors = tensor
        if cols_to_output_tensors is not None:
          cols_to_output_tensors[c] = processed_tensors
        non_coalesced_column_output_tensors.append(
            tuple([cid, processed_tensors]))

    coalesced_column_output_tensors = []
    group_columns = []
    group_inputs = []
    group_weights = []
    for cid, c in indexed_coalesced_columns:
      with ops.name_scope(c.name):
        group_columns.append(c)
        sparse_tensors = c.categorical_column.get_sparse_tensors(
            transformation_cache, self._state_manager)
        group_inputs.append(sparse_tensors)
        if hasattr(c, 'shared_embedding_column_creator'):
          group_weights.append(
              c.shared_embedding_column_creator.embedding_weights)
        else:
          group_weights.append(
              self._state_manager.get_variable(c, self._impl.weight_name(c)))
    _, coalesced_columns = zip(*indexed_coalesced_columns)
    group_outputs = self._lookup_sparse_coalesced(
        group_weights, group_inputs, coalesced_columns,
        num_groups=self.num_groups)
    for idx, cid_and_column in enumerate(indexed_coalesced_columns):
      cid, c = cid_and_column
      with ops.name_scope(c.name):
        if hasattr(self, '_process_dense_tensor'):
          processed_tensors = self._process_dense_tensor(c, group_outputs[idx])
        else:
          processed_tensors = group_outputs[idx]
        if cols_to_output_tensors is not None:
          cols_to_output_tensors[c] = processed_tensors
        coalesced_column_output_tensors.append(
            tuple([cid, processed_tensors]))

    indexed_output_tensors = sorted(
        coalesced_column_output_tensors + non_coalesced_column_output_tensors)
    _, output_tensors = zip(*indexed_output_tensors)

    if not self._enable_concat:
      return output_tensors

    if hasattr(self, '_verify_and_concat_tensors'):
      return self._verify_and_concat_tensors(output_tensors)
    return array_ops.concat(output_tensors, 1)

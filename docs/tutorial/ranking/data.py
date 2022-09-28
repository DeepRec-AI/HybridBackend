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

r'''Data utilities for ranking model.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json

import tensorflow as tf

EmbeddingSpec = collections.namedtuple(
  'EmbeddingSpec', ['size', 'dimension'])


FeatureSpec = collections.namedtuple(
  'FeatureSpec',
  ['name', 'dtype', 'type', 'default', 'norm', 'log', 'embedding'])


class DataSpec(object):  # pylint: disable=useless-object-inheritance
  r'''Data specification.
  '''
  class Types(object):  # pylint: disable=useless-object-inheritance
    r'''Data column types.
    '''
    SCALAR = 'scalar'
    LIST = 'list'

  @classmethod
  def read(
      cls, path,
      disable_imputation=False,
      disable_transform=False,
      override_embedding_size=None):
    r'''Read from path.

    Args:
      path: File path of data spec file.
      disable_imputation: Do not handle missing values.
      disable_transform: Do not transform numeric values.
      override_embedding_size: Use specific size not configued size.
    '''
    with open(path, encoding='utf8') as f:
      specs = json.load(f)
    return cls(
      specs,
      disable_imputation=disable_imputation,
      disable_transform=disable_transform,
      override_embedding_size=override_embedding_size)

  def __init__(
      self, items,
      disable_imputation=False,
      disable_transform=False,
      override_embedding_size=None):
    r'''Constructor.

    Args:
      items: list of data spec items.
      disable_imputation: Do not handle missing values.
      disable_transform: Do not transform numeric values.
      override_embedding_size: Use specific size not configued size.
    '''
    if not isinstance(items, (list, tuple)):
      raise ValueError('items must be a list')
    self._feature_specs = []
    for item in items:
      embedding = None
      if 'embedding' in item:
        embedding = EmbeddingSpec(
          item['embedding']['size'],
          item['embedding']['dimension'])
      self._feature_specs.append(
        FeatureSpec(
          item['name'],
          item['dtype'],
          item['type'],
          item['default'],
          item['norm'] if 'norm' in item else None,
          item['log'] if 'log' in item else None,
          embedding))
    self._disable_imputation = disable_imputation
    self._disable_transform = disable_transform
    self._override_embedding_size = override_embedding_size

  def __iter__(self):
    return iter(self._feature_specs)

  def to_example_spec(self):
    r'''Generate example spec for TFRecord dataset.
    '''
    return {
      **{
        f.name: tf.io.FixedLenFeature([], f.dtype)
        for f in self._feature_specs if f.type == DataSpec.Types.SCALAR},
      **{
        f.name: tf.io.VarLenFeature(f.dtype)
        for f in self._feature_specs if f.type == DataSpec.Types.LIST}}

  @property
  def feature_specs(self):
    return self._feature_specs

  @property
  def defaults(self):
    return {c.name: c.default for c in self._feature_specs}

  @property
  def norms(self):
    return {c.name: c.norm for c in self._feature_specs}

  @property
  def logs(self):
    return {c.name: c.log for c in self._feature_specs}

  @property
  def embedding_dims(self):
    return {
      c.name: c.embedding.dimension if c.embedding else None
      for c in self._feature_specs}

  @property
  def embedding_sizes(self):
    return {
      c.name: c.embedding.size if c.embedding else None
      for c in self._feature_specs}

  def transform_numeric(self, field, feature):
    r'''Transform numeric features.
    '''
    if feature.dtype != tf.float32:
      default_value = None if self._disable_imputation else self.defaults[field]
      if default_value is not None:
        is_valid = tf.greater_equal(feature, 0)
        numeric_defaults = tf.zeros_like(feature) + default_value
        feature = tf.where(is_valid, feature, numeric_defaults)
      feature = tf.to_float(feature)
    log = None if self._disable_transform else self.logs[field]
    if log is not None:
      feature = tf.math.log1p(feature)
    norm = None if self._disable_transform else self.norms[field]
    if norm is not None:
      feature = feature / norm
    return tf.reshape(feature, shape=[-1, 1])

  def transform_categorical(self, field, feature, embedding_weights):
    r'''Transform categorical features.
    '''
    default_value = None if self._disable_imputation else self.defaults[field]
    embedding_size = (
      self._override_embedding_size if self._override_embedding_size
      else self.embedding_sizes[field])

    if isinstance(feature, tf.Tensor):
      if default_value is not None:
        is_valid = tf.greater_equal(feature, 0)
        id_defaults = tf.zeros_like(feature) + default_value
        feature = tf.where(is_valid, feature, id_defaults)
      feature = feature % embedding_size
      feature, idx = tf.unique(tf.reshape(feature, shape=[-1]))
      embedding = tf.nn.embedding_lookup(embedding_weights, feature)
      return tf.gather(embedding, idx)

    feature = tf.SparseTensor(
      feature.indices,
      feature.values % embedding_size,
      feature.dense_shape)
    if default_value is None:
      return tf.nn.embedding_lookup_sparse(embedding_weights, feature, None)

    return tf.nn.safe_embedding_lookup_sparse(
      embedding_weights, feature,
      default_id=default_value)

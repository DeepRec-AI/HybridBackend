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

r'''Layers for ranking model.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


class DotInteract(tf.layers.Layer):
  r'''DLRM: Deep Learning Recommendation Model for Personalization and
  Recommendation Systems.

  See https://github.com/facebookresearch/dlrm for more information.
  '''
  def call(self, x):
    r'''Call the DLRM dot interact layer.
    '''
    x2 = tf.matmul(x, x, transpose_b=True)
    x2_dim = x2.shape[-1]
    x2_ones = tf.ones_like(x2)
    x2_mask = tf.linalg.band_part(x2_ones, 0, -1)
    y = tf.boolean_mask(x2, x2_ones - x2_mask)
    y = tf.reshape(y, [-1, x2_dim * (x2_dim - 1) // 2])
    return y


class Cross(tf.layers.Layer):
  r'''DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale
  Learning to Rank Systems.

  See https://arxiv.org/abs/2008.13535 for more information.
  '''
  def call(self, x):
    r'''Call the DCN cross layer.
    '''
    x2 = tf.layers.dense(
      x, x.shape[-1],
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(),
      bias_initializer=tf.zeros_initializer())
    y = x * x2 + x
    y = tf.reshape(y, [-1, x.shape[1] * x.shape[2]])
    return y


class Ranking(tf.layers.Layer):
  r'''A simple ranking model.
  '''
  def __init__(
      self,
      embedding_columns,
      bottom_mlp=None,
      top_mlp=None,
      feature_interaction=None,
      **kwargs):
    r'''Constructor.

    Args:
      embedding_columns: List of embedding columns.
      bottom_mlp: List of bottom MLP dimensions.
      top_mlp: List of top MLP dimensions.
      feature_interaction: Feature interaction layer class.
      **kwargs: keyword named properties.
    '''
    super().__init__(**kwargs)

    if bottom_mlp is None:
      bottom_mlp = [512, 256, 64]
    self.bottom_mlp = bottom_mlp
    if top_mlp is None:
      top_mlp = [1024, 1024, 512, 256, 1]
    self.top_mlp = top_mlp
    if feature_interaction is None:
      feature_interaction = DotInteract
    self.feature_interaction = feature_interaction
    self.embedding_columns = embedding_columns
    dimensions = {c.dimension for c in embedding_columns}
    if len(dimensions) > 1:
      raise ValueError('Only one dimension supported')
    self.dimension = list(dimensions)[0]

  def call(self, values, embeddings):
    r'''Call the dlrm model
    '''
    with tf.name_scope('bottom_mlp'):
      bot_mlp_input = tf.math.log(values + 1.)
      for i, d in enumerate(self.bottom_mlp):
        bot_mlp_input = tf.layers.dense(
          bot_mlp_input, d,
          activation=tf.nn.relu,
          kernel_initializer=tf.glorot_normal_initializer(),
          bias_initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=math.sqrt(1.0 / d)),
          name=f'bottom_mlp_{i}')
      bot_mlp_output = tf.layers.dense(
        bot_mlp_input, self.dimension,
        activation=tf.nn.relu,
        kernel_initializer=tf.glorot_normal_initializer(),
        bias_initializer=tf.random_normal_initializer(
          mean=0.0,
          stddev=math.sqrt(1.0 / self.dimension)),
        name='bottom_mlp_output')

    with tf.name_scope('feature_interaction'):
      feat_interact_input = tf.concat([bot_mlp_output] + embeddings, axis=-1)
      feat_interact_input = tf.reshape(
        feat_interact_input,
        [-1, 1 + len(embeddings), self.dimension])
      feat_interact_output = self.feature_interaction()(feat_interact_input)

    with tf.name_scope('top_mlp'):
      top_mlp_input = tf.concat([bot_mlp_output, feat_interact_output], axis=1)
      num_fields = len(self.embedding_columns)
      prev_d = (num_fields * (num_fields + 1)) / 2 + self.dimension
      for i, d in enumerate(self.top_mlp[:-1]):
        top_mlp_input = tf.layers.dense(
          top_mlp_input, d,
          activation=tf.nn.relu,
          kernel_initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=math.sqrt(2.0 / (prev_d + d))),
          bias_initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=math.sqrt(1.0 / d)),
          name=f'top_mlp_{i}')
        prev_d = d
      top_mlp_output = tf.layers.dense(
        top_mlp_input, self.top_mlp[-1],
        activation=tf.nn.sigmoid,
        kernel_initializer=tf.random_normal_initializer(
          mean=0.0,
          stddev=math.sqrt(2.0 / (prev_d + self.top_mlp[-1]))),
        bias_initializer=tf.random_normal_initializer(
          mean=0.0,
          stddev=math.sqrt(1.0 / self.top_mlp[-1])),
        name=f'top_mlp_{len(self.top_mlp) - 1}')
    return top_mlp_output

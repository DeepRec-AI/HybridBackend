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

r'''Utilities for ranking model.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def stacked_dcn_v2(features, mlp_dims):
  r'''Stacked DCNv2.

  DCNv2: Improved Deep & Cross Network and Practical Lessons for Web-scale
  Learning to Rank Systems.

  See https://arxiv.org/abs/2008.13535 for more information.
  '''
  with tf.name_scope('cross'):
    cross_input = tf.concat(features, axis=-1)
    cross_input_shape = [-1, sum([f.shape[-1] for f in features])]
    cross_input = tf.reshape(cross_input, cross_input_shape)
    cross_input_sq = tf.layers.dense(
      cross_input, cross_input.shape[-1],
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(),
      bias_initializer=tf.zeros_initializer())
    cross_output = cross_input * cross_input_sq + cross_input
    cross_output = tf.reshape(cross_output, [-1, cross_input.shape[1]])
    cross_output_dim = (len(features) * (len(features) + 1)) / 2

  with tf.name_scope('mlp'):
    prev_layer = cross_output
    prev_dim = cross_output_dim
    for i, d in enumerate(mlp_dims[:-1]):
      prev_layer = tf.layers.dense(
        prev_layer, d,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(
          mean=0.0,
          stddev=math.sqrt(2.0 / (prev_dim + d))),
        bias_initializer=tf.random_normal_initializer(
          mean=0.0,
          stddev=math.sqrt(1.0 / d)),
        name=f'mlp_{i}')
      prev_dim = d
    return tf.layers.dense(
      prev_layer, mlp_dims[-1],
      activation=tf.nn.sigmoid,
      kernel_initializer=tf.random_normal_initializer(
        mean=0.0,
        stddev=math.sqrt(2.0 / (prev_dim + mlp_dims[-1]))),
      bias_initializer=tf.random_normal_initializer(
        mean=0.0,
        stddev=math.sqrt(1.0 / mlp_dims[-1])),
      name=f'mlp_{len(mlp_dims) - 1}')


def dlrm(
    wide_features, deep_features,
    bottom_mlp_dims, dot_interact_dim, top_mlp_dims):
  r'''DLRM.

  DLRM: Deep Learning Recommendation Model for Personalization and
  Recommendation Systems.

  See https://github.com/facebookresearch/dlrm for more information.
  '''
  with tf.name_scope('bottom_mlp'):
    bottom_mlp_input = tf.concat(wide_features, axis=1)
    bottom_mlp_input = tf.math.log1p(tf.to_float(bottom_mlp_input))
    for i, d in enumerate(bottom_mlp_dims):
      bottom_mlp_input = tf.layers.dense(
        bottom_mlp_input, d,
        activation=tf.nn.relu,
        kernel_initializer=tf.glorot_normal_initializer(),
        bias_initializer=tf.random_normal_initializer(
          mean=0.0,
          stddev=math.sqrt(1.0 / d)),
        name=f'bottom_mlp_{i}')
    bottom_mlp_output = tf.layers.dense(
      bottom_mlp_input, dot_interact_dim,
      activation=tf.nn.relu,
      kernel_initializer=tf.glorot_normal_initializer(),
      bias_initializer=tf.random_normal_initializer(
        mean=0.0,
        stddev=math.sqrt(1.0 / dot_interact_dim)),
      name='bottom_mlp_output')

  with tf.name_scope('dot_interact'):
    dot_interact_len = len(deep_features)
    dot_interact_input = tf.concat([bottom_mlp_output] + deep_features, axis=-1)
    dot_interact_input = tf.reshape(
      dot_interact_input,
      [-1, dot_interact_len + 1, dot_interact_dim])
    x2 = tf.matmul(dot_interact_input, dot_interact_input, transpose_b=True)
    x2_dim = x2.shape[-1]
    x2_ones = tf.ones_like(x2)
    x2_mask = tf.linalg.band_part(x2_ones, 0, -1)
    dot_interact_output = tf.boolean_mask(x2, x2_ones - x2_mask)
    dot_interact_output = tf.reshape(
      dot_interact_output, [-1, x2_dim * (x2_dim - 1) // 2])

  with tf.name_scope('top_mlp'):
    top_mlp_input = tf.concat([bottom_mlp_output, dot_interact_output], axis=1)
    prev_d = (dot_interact_len * (dot_interact_len + 1)) / 2 + dot_interact_dim
    for i, d in enumerate(top_mlp_dims[:-1]):
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
    return tf.layers.dense(
      top_mlp_input, top_mlp_dims[-1],
      activation=tf.nn.sigmoid,
      kernel_initializer=tf.random_normal_initializer(
        mean=0.0,
        stddev=math.sqrt(2.0 / (prev_d + top_mlp_dims[-1]))),
      bias_initializer=tf.random_normal_initializer(
        mean=0.0,
        stddev=math.sqrt(1.0 / top_mlp_dims[-1])),
      name=f'top_mlp_{len(top_mlp_dims) - 1}')

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

r'''Tests for `hb.train.wraps_optimizer`.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np

import hybridbackend.test as hbtest


def _test_sgd(_, lr):
  r'''Test SGD.
  '''
  # pylint: disable=import-outside-toplevel
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default():
    with hb.scope():
      v = tf.get_variable(
        'v0',
        initializer=tf.random_normal([4, 3], mean=0.1, stddev=0.06))
      loss = -tf.reduce_sum(v + 3.14)
      opt = tf.train.GradientDescentOptimizer(lr)
      train_op = opt.minimize(loss)
      steps = []
      with hb.train.monitored_session() as sess:
        steps.append(sess.run(v))
        steps.append(sess.run(train_op))
        steps.append(sess.run(v))
      return steps


def _test_adam(_, lr):
  r'''Test Adam.
  '''
  # pylint: disable=import-outside-toplevel
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default():
    with hb.scope():
      v = tf.get_variable(
        'v0',
        initializer=tf.random_normal([4, 3], mean=0.1, stddev=0.06))
      loss = -tf.reduce_sum(v + 3.14)
      opt = tf.train.AdamOptimizer(lr)
      train_op = opt.minimize(loss)
      steps = []
      with hb.train.monitored_session() as sess:
        steps.append(sess.run(v))
        steps.append(sess.run(train_op))
        steps.append(sess.run(v))
      return steps


def _test_adam_function(_, lr):
  r'''Test Adam using function decorator.
  '''
  # pylint: disable=import-outside-toplevel
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  @hb.function()
  def model_fn():
    v = tf.get_variable(
      'v0', initializer=tf.random_normal([4, 3], mean=0.1, stddev=0.06))
    loss = -tf.reduce_sum(v + 3.14)
    opt = tf.train.AdamOptimizer(lr)
    return v, opt.minimize(loss)

  with tf.Graph().as_default():
    v, train_op = model_fn()
    steps = []
    with hb.train.monitored_session() as sess:
      steps.append(sess.run(v))
      steps.append(sess.run(train_op))
      steps.append(sess.run(v))
      return steps


@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
class OptimizerTest(unittest.TestCase):
  r'''Tests for `wraps_optimizer`.
  '''
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

  def test_sgd(self):
    lr = 1.0
    results = hbtest.Spawn(2)(lambda rank: _test_sgd(rank, lr))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-4)
    np.testing.assert_allclose(results[0][2], results[1][2], atol=1e-4)
    np.testing.assert_allclose(results[0][2], results[0][0] + lr)

  def test_adam(self):
    lr = 1.0
    results = hbtest.Spawn(2)(lambda rank: _test_adam(rank, lr))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-4)
    np.testing.assert_allclose(results[0][2], results[1][2], atol=1e-4)

  def test_adam_function(self):
    lr = 1.0
    results = hbtest.Spawn(2)(lambda rank: _test_adam_function(rank, lr))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-4)
    np.testing.assert_allclose(results[0][2], results[1][2], atol=1e-4)


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

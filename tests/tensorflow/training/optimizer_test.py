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

import numpy as np
import os
import unittest

from tests.tensorflow.spawn import Spawn
from tests.tensorflow.spawn import register


def _test_sgd(_, lr):
  r'''Test SGD.
  '''
  # pylint: disable=import-outside-toplevel
  import tensorflow as tf

  from hybridbackend.tensorflow.training.optimizer import wraps_optimizer
  from hybridbackend.tensorflow.training.server_lib import device_setter
  from hybridbackend.tensorflow.training.server import Server

  with tf.Graph().as_default():
    server = Server()
    with tf.device(device_setter()):
      v = tf.get_variable(
        'v0',
        initializer=tf.random_normal([4, 3], mean=0.1, stddev=0.06))
      loss = -tf.reduce_sum(v + 3.14)
      opt = wraps_optimizer(tf.train.GradientDescentOptimizer)(lr)
      train_op = opt.minimize(loss)
      steps = []
      with tf.train.MonitoredTrainingSession(
          server.target,
          is_chief=True,
          hooks=[opt.make_session_run_hook()]) as sess:
        steps.append(sess.run(v))
        steps.append(sess.run(train_op))
        steps.append(sess.run(v))
      return steps


def _test_adam(_, lr):
  r'''Test Adam.
  '''
  # pylint: disable=import-outside-toplevel
  import tensorflow as tf

  from hybridbackend.tensorflow.training.optimizer import wraps_optimizer
  from hybridbackend.tensorflow.training.server_lib import device_setter
  from hybridbackend.tensorflow.training.server import Server

  with tf.Graph().as_default():
    server = Server()
    with tf.device(device_setter()):
      v = tf.get_variable(
        'v0',
        initializer=tf.random_normal([4, 3], mean=0.1, stddev=0.06))
      loss = -tf.reduce_sum(v + 3.14)
      opt = wraps_optimizer(tf.train.AdamOptimizer)(lr)
      train_op = opt.minimize(loss)
      steps = []
      with tf.train.MonitoredTrainingSession(
          server.target,
          is_chief=True,
          hooks=[opt.make_session_run_hook()]) as sess:
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

  def model_fn():
    v = tf.get_variable(
      'v0', initializer=tf.random_normal([4, 3], mean=0.1, stddev=0.06))
    loss = -tf.reduce_sum(v + 3.14)
    opt = hb.train.AdamOptimizer(lr)
    return v, opt.minimize(loss)

  with tf.Graph().as_default():
    server = hb.train.Server()
    with tf.device(hb.train.device_setter()):
      v, train_op = model_fn()
    steps = []
    with hb.train.MonitoredTrainingSession(
        server.target,
        is_chief=True) as sess:
      steps.append(sess.run(v))
      steps.append(sess.run(train_op))
      steps.append(sess.run(v))
      return steps


class OptimizerTest(unittest.TestCase):
  r'''Tests for `wraps_optimizer`.
  '''
  def test_sgd(self):
    lr = 1.0
    results = Spawn(2)(lambda rank: _test_sgd(rank, lr))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-4)
    np.testing.assert_allclose(results[0][2], results[1][2], atol=1e-4)
    np.testing.assert_allclose(results[0][2], results[0][0] + lr)

  def test_adam(self):
    lr = 1.0
    results = Spawn(2)(lambda rank: _test_adam(rank, lr))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-4)
    np.testing.assert_allclose(results[0][2], results[1][2], atol=1e-4)

  def test_adam_function(self):
    lr = 1.0
    results = Spawn(2)(lambda rank: _test_adam_function(rank, lr))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-4)
    np.testing.assert_allclose(results[0][2], results[1][2], atol=1e-4)


if __name__ == '__main__':
  register(['gpu', 'train'])
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  unittest.main()

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

r'''Tests for allreduce collective communication.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np

import hybridbackend.common.test as hbtest

# pylint: disable=missing-docstring,import-outside-toplevel


def _test_allreduce(rank, a, b):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default():
    with hb.scope():
      input0 = tf.constant(a) if rank == 0 else tf.constant(b)
      sum0 = hb.distribute.allreduce(input0)
      with tf.train.MonitoredTrainingSession('') as sess:
        return sess.run(sum0)


def _test_allreduce_max(rank, a, b):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default():
    with hb.scope():
      input0 = tf.constant(a) if rank == 0 else tf.constant(b)
      sum0 = hb.distribute.allreduce(
        input0, reduce_op=hb.distribute.ops.MAX)
      with tf.train.MonitoredTrainingSession('') as sess:
        return sess.run(sum0)


def _test_allreduce_n(rank, a, b):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default():
    with hb.scope():
      input0 = tf.constant(a) if rank == 0 else tf.constant(b)
      sum0 = hb.distribute.allreduce(input0)
      sum1 = hb.distribute.allreduce(input0)
      sum2 = hb.distribute.allreduce(sum0)
      result = sum1 + sum2
      with tf.train.MonitoredTrainingSession('') as sess:
        return sess.run(result)


def _test_allreduce_grad(rank, g, shp, a, b):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default():
    with hb.scope():
      data = tf.constant(a if rank == 0 else b, shape=shp)
      reduced = hb.distribute.allreduce(data)
      loss = tf.reduce_mean(reduced)
      grad = tf.gradients(
        [loss], [data], [g], colocate_gradients_with_ops=True)
      with tf.train.MonitoredTrainingSession('') as sess:
        return sess.run(grad)


@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_NCCL') == 'ON', 'NCCL required')
class AllreduceTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['TF_CPP_VMODULE'] = (
      'nccl_comm=1,'
      'nccl_create=1,'
      'nccl_allreduce=1')

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_allreduce(self):
    a = 13
    b = 22
    results = hbtest.Spawn(2)(
      lambda rank: _test_allreduce(rank, a, b))
    np.testing.assert_allclose(results[0], a + b, rtol=1e-6)
    np.testing.assert_allclose(results[1], a + b, rtol=1e-6)

  def test_allreduce_fallback(self):
    a = 13
    results = hbtest.Spawn(1)(
      lambda rank: _test_allreduce(rank, a, 22))
    np.testing.assert_allclose(results[0], a, rtol=1e-6)

  def xtest_allreduce_max(self):
    a = 13
    b = 22
    results = hbtest.Spawn(2)(
      lambda rank: _test_allreduce_max(rank, a, b))
    np.testing.assert_allclose(results[0], b, rtol=1e-6)
    np.testing.assert_allclose(results[1], b, rtol=1e-6)

  def test_allreduce_n(self):
    a = 13
    b = 22
    results = hbtest.Spawn(2)(
      lambda rank: _test_allreduce_n(rank, a, b))
    np.testing.assert_allclose(results[0], 3 * (a + b), rtol=1e-6)
    np.testing.assert_allclose(results[1], 3 * (a + b), rtol=1e-6)

  def test_allreduce_grad(self):
    g = 2.0
    w = 4
    h = 8
    results = hbtest.Spawn(2)(
      lambda rank: _test_allreduce_grad(rank, g, [w, h], 1.0, 2.0))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6)
    np.testing.assert_allclose(
      results[0],
      [[[g / (w * h) * 2] * h] * w], rtol=1e-6)


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

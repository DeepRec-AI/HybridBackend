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

r'''Tests for ReduceScatter.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import hybridbackend.tensorflow as hb
import hybridbackend.test as hbtest
import unittest


# pylint: disable=missing-docstring
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
class ReduceScatterTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['TF_CPP_VMODULE'] = 'nccl_reduce_scatter=1'

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def _test_reduce_scatter(self, devices, shapes, islegal=True):
    hb.context.options.update(comm_pubsub_device='')

    num_comms = len(shapes)
    num_devices = len(devices)
    all_inputs = []
    all_sums = []

    with tf.Graph().as_default():
      for i in xrange(num_comms):
        comm_inputs = []
        comm_sums = []
        shared_name = f'comm_{i}'
        for d in xrange(num_devices):
          with tf.device(devices[d]):
            comm_inputs.append(
              tf.get_variable(
                f'input{i}/part_{d}',
                initializer=tf.random_normal(
                  shapes[i], mean=100, stddev=80)))
            comm = hb.distribute.Communicator.build(shared_name, devices)
            comm_sums.append(comm.reduce_scatter(comm_inputs[d]))
        all_inputs.append(comm_inputs)
        all_sums.append(comm_sums)
      baselines = (
        [tf.split(tf.add_n(all_inputs[i]), num_devices, axis=0)
         for i in xrange(num_comms)]
        if islegal else [])
      with hb.train.monitored_session() as sess:
        expects = sess.run(baselines)
        actuals = sess.run(all_sums)
      for i in xrange(num_comms):
        for d in xrange(num_devices):
          np.testing.assert_allclose(actuals[i][d], expects[i][d], atol=1e-4)

  def test_one_device_one_tensor_reduce_scatter(self):
    self._test_reduce_scatter(
      [f'/job:localhost/replica:0/task:0/device:GPU:{d}' for d in [0]],
      [[16, 4]])

  def test_two_devices_scalar_tensors_reduce_scatter(self):
    with self.assertRaises(ValueError):
      self._test_reduce_scatter(
        [f'/job:localhost/replica:0/task:0/device:GPU:{d}' for d in [0, 1]],
        [[]])
      self.fail('Should throw ValueError here.')

  def test_two_devices_four_tensors_reduce_scatter(self):
    self._test_reduce_scatter(
      [f'/job:localhost/replica:0/task:0/device:GPU:{d}' for d in [0, 1]],
      [[18, 41, 37], [16, 121], [2], [16, 6]])

  def test_two_devices_six_tensors_reduce_scatter(self):
    self._test_reduce_scatter(
      [f'/job:localhost/replica:0/task:0/device:GPU:{d}'
       for d in [0, 1]],
      [[16, 4], [102, 100], [4, 5, 8], [12], [6, 2, 2, 3], [1048]])

  def test_two_devices_irregular_reduce_scatter(self):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self._test_reduce_scatter(
        [f'/job:localhost/replica:0/task:0/device:GPU:{d}' for d in [0, 1]],
        [[17, 41, 37]], False)
      self.fail('Should throw InvalidArgumentError here.')

  def test_reduce_scatter_grad(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    shared_name = 'comm'
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        a = tf.constant(1.0, shape=[4, 8])
        comm0 = hb.distribute.Communicator.build(shared_name, devices)
        recv0 = comm0.reduce_scatter(a * 0.5)
        loss0 = tf.reduce_mean(recv0) * 20.0
      with tf.device('/gpu:1'):
        b = tf.constant(2.0, shape=[4, 8])
        comm1 = hb.distribute.Communicator.build(shared_name, devices)
        recv1 = comm1.reduce_scatter(b * 0.75)
        loss1 = tf.reduce_mean(recv1) * 10.0
      loss = loss0 * loss1 + loss1 * 2.0
      grad0, grad1 = tf.gradients([loss], [a, b], [2.0])
      with hb.train.monitored_session() as sess:
        g0, g1 = sess.run([grad0, grad1])
        np.testing.assert_allclose(
          g0,
          [[25.0] * 8, [25.0] * 8, [26.25] * 8, [26.25] * 8],
          rtol=1e-6)
        np.testing.assert_allclose(
          g1,
          [[37.5] * 8, [37.5] * 8, [39.375] * 8, [39.375] * 8],
          rtol=1e-6)

  def test_reduce_scatter_grad_error(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    shared_name = 'comm'
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        a = tf.constant(1.0, shape=[4, 8])
        comm0 = hb.distribute.Communicator.build(shared_name, devices)
        recv0 = comm0.reduce_scatter(a * 0.5, reduce_op=hb.distribute.ops.MAX)
        loss0 = tf.reduce_mean(recv0) * 20.0
      with tf.device('/gpu:1'):
        b = tf.constant(2.0, shape=[4, 8])
        comm1 = hb.distribute.Communicator.build(shared_name, devices)
        recv1 = comm1.reduce_scatter(b * 0.75, reduce_op=hb.distribute.ops.MAX)
        loss1 = tf.reduce_mean(recv1) * 10.0
      loss = loss0 * loss1 + loss1 * 2.0
      with self.assertRaises(RuntimeError):
        tf.gradients([loss], [a, b], [2.0], colocate_gradients_with_ops=True)


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

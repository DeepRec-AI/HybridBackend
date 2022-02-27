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

r'''Tests for Reduce.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
import unittest

from hybridbackend.tensorflow.distribute.communicator import Communicator
from hybridbackend.tensorflow.framework.context import context
from hybridbackend.tensorflow.framework.ops import CollectiveOps
from hybridbackend.tensorflow.training.server import MonitoredTrainingSession
from hybridbackend.tensorflow.training.server import Server

from tests.tensorflow.spawn import register


# pylint: disable=missing-docstring
class ReduceTest(unittest.TestCase):
  def _simple_reduce(self, root_rank=0):
    context.update_params(pubsub_device='')

    a = 13
    b = 22
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        comm0 = Communicator.create(shared_name, devices)
        input0 = tf.constant(a)
        sum0 = comm0.reduce(input0, root_rank=root_rank)
      with tf.device('/gpu:1'):
        comm1 = Communicator.create(shared_name, devices)
        input1 = tf.constant(b)
        sum1 = comm1.reduce(input1, root_rank=root_rank)
      with MonitoredTrainingSession('', is_chief=True) as sess:
        s0, s1 = sess.run([sum0, sum1])
        if root_rank == 0:
          np.testing.assert_allclose(s0, a + b, rtol=1e-6)
        else:
          np.testing.assert_allclose(s1, a + b, rtol=1e-6)

  def test_simple_reduce(self):
    self._simple_reduce(root_rank=0)
    self._simple_reduce(root_rank=1)

  def test_onedevice_reduce(self):
    context.update_params(pubsub_device='')

    a = 13
    devices = ['/gpu:0']
    shared_name = "comm_onedevice"
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        comm0 = Communicator.create(shared_name, devices)
        input0 = tf.constant(a)
        sum0 = comm0.reduce(input0)
      with MonitoredTrainingSession('', is_chief=True) as sess:
        np.testing.assert_allclose(sess.run(sum0), a, rtol=1e-6)

  def test_reduce_max(self):
    context.update_params(pubsub_device='')

    a = 13
    b = 22
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        comm0 = Communicator.create(shared_name, devices)
        input0 = tf.constant(a)
        max0 = comm0.reduce(input0, reduce_op=CollectiveOps.MAX)
      with tf.device('/gpu:1'):
        comm1 = Communicator.create(shared_name, devices)
        input1 = tf.constant(b)
        max1 = comm1.reduce(input1, reduce_op=CollectiveOps.MAX)
      with MonitoredTrainingSession('', is_chief=True) as sess:
        s0, _ = sess.run([max0, max1])
        np.testing.assert_allclose(s0, b, rtol=1e-6)

  def test_multicomm(self):
    context.update_params(pubsub_device='')

    a = 13
    b = 22
    devices = ['/gpu:0', '/gpu:1']
    with tf.Graph().as_default():
      reduce_ops_rank0 = []
      reduce_ops_rank1 = []
      for icomm in xrange(10):
        comm_name = f'comm{icomm}'
        with tf.device('/gpu:0'):
          comm0 = Communicator.create(comm_name, devices)
          input0 = tf.constant(a)
          sum0 = comm0.reduce(input0)
        with tf.device('/gpu:1'):
          comm1 = Communicator.create(comm_name, devices)
          input1 = tf.constant(b)
          sum1 = comm1.reduce(input1)
        reduce_ops_rank0.append(sum0)
        reduce_ops_rank1.append(sum1)
      with MonitoredTrainingSession('', is_chief=True) as sess:
        results_rank0, _ = sess.run([reduce_ops_rank0, reduce_ops_rank1])
        for r in results_rank0:
          np.testing.assert_allclose(r, a + b, rtol=1e-6)

  def test_monitored_training_session(self):
    context.update_params(pubsub_device='')

    a = 13
    b = 22
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    server = Server({"localhost": ["localhost:0"]})
    try:
      with tf.Graph().as_default():
        with tf.device('/gpu:0'):
          comm0 = Communicator.create(shared_name, devices)
          input0 = tf.constant(a)
          sum0 = comm0.reduce(input0)
        with tf.device('/gpu:1'):
          comm1 = Communicator.create(shared_name, devices)
          input1 = tf.constant(b)
          sum1 = comm1.reduce(input1)

        with MonitoredTrainingSession(server.target) as sess:
          s0, _ = sess.run([sum0, sum1])
          np.testing.assert_allclose(s0, a + b, rtol=1e-6)
    finally:
      del server

  def test_reduce_grad(self):
    context.update_params(pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        a = tf.constant(1.0, shape=[2, 10])
        comm0 = Communicator.create(shared_name, devices)
        recv0 = comm0.reduce(a * 0.75)
        loss = tf.reduce_mean(recv0) * 20.0
      with tf.device('/gpu:1'):
        b = tf.constant(2.0, shape=[2, 10])
        comm1 = Communicator.create(shared_name, devices)
        recv1 = comm1.reduce(b)
      grad = tf.gradients(
          [loss], [a], [2.0], colocate_gradients_with_ops=True)
      with tf.train.MonitoredTrainingSession('', is_chief=True) as sess:
        g, _ = sess.run([grad, recv1])
        np.testing.assert_allclose(g[0], [[1.5]*10]*2, rtol=1e-6)

  def test_reduce_grad_error(self):
    context.update_params(pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        a = tf.constant(1.0, shape=[2, 10])
        comm0 = Communicator.create(shared_name, devices)
        recv0 = comm0.reduce(a * 0.75, reduce_op=CollectiveOps.MAX)
        loss = tf.reduce_mean(recv0) * 20.0
      with tf.device('/gpu:1'):
        b = tf.constant(2.0, shape=[2, 10])
        comm1 = Communicator.create(shared_name, devices)
        _ = comm1.reduce(b, reduce_op=CollectiveOps.MAX)
      with self.assertRaises(RuntimeError):
        tf.gradients([loss], [a], [2.0])


if __name__ == '__main__':
  register(['gpu', 'dist'])
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  os.environ['TF_CPP_VMODULE'] = 'nccl_reduce=1'
  unittest.main()

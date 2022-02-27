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

r'''Tests for Allreduce gradients.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import unittest

import tensorflow as tf
from hybridbackend.tensorflow.distribute.communicator import Communicator
from hybridbackend.tensorflow.framework.context import context
from hybridbackend.tensorflow.framework.ops import CollectiveOps
from hybridbackend.tensorflow.training.server import MonitoredTrainingSession

from tests.tensorflow.spawn import register


# pylint: disable=missing-docstring
class AllreduceGradTest(unittest.TestCase):
  def test_allreduce_grad(self):
    context.update_params(pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        a = tf.constant(1.0, shape=[2, 10])
        comm0 = Communicator.create(shared_name, devices)
        recv0 = comm0.allreduce(a * 0.75)
        loss0 = tf.reduce_mean(recv0) * 20.0
      with tf.device('/gpu:1'):
        b = tf.constant(2.0, shape=[2, 10])
        comm1 = Communicator.create(shared_name, devices)
        recv1 = comm1.allreduce(b)
        loss1 = tf.reduce_mean(recv1) * 10.0
      loss = loss0 * loss1
      grad0, grad1 = tf.gradients(
          [loss], [a, b], [2.0], colocate_gradients_with_ops=True)
      with MonitoredTrainingSession('', is_chief=True) as sess:
        g0, g1 = sess.run([grad0, grad1])
        np.testing.assert_allclose(
            g0,
            [[82.5]*10]*2,
            rtol=1e-6)
        np.testing.assert_allclose(
            g1,
            [[110.0]*10]*2,
            rtol=1e-6)

  def test_allreduce_grad_error(self):
    context.update_params(pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        a = tf.constant(1.0, shape=[2, 10])
        comm0 = Communicator.create(shared_name, devices)
        recv0 = comm0.allreduce(a * 0.75, reduce_op=CollectiveOps.MAX)
        loss0 = tf.reduce_mean(recv0) * 20.0
      with tf.device('/gpu:1'):
        b = tf.constant(2.0, shape=[2, 10])
        comm1 = Communicator.create(shared_name, devices)
        recv1 = comm1.allreduce(b, reduce_op=CollectiveOps.MAX)
        loss1 = tf.reduce_mean(recv1) * 10.0
      loss = loss0 * loss1
      with self.assertRaises(RuntimeError):
        tf.gradients([loss], [a, b], [2.0])


if __name__ == '__main__':
  register(['gpu', 'dist'])
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  os.environ['TF_CPP_VMODULE'] = 'nccl_allreduce=1'
  unittest.main()

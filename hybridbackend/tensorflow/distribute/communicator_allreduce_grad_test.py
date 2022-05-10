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
import tensorflow as tf

import hybridbackend.tensorflow as hb
import hybridbackend.test as hbtest
import unittest


# pylint: disable=missing-docstring
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
class AllreduceGradTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['TF_CPP_VMODULE'] = 'nccl_allreduce=1'

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_allreduce_grad(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    shared_name = 'comm'
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        a = tf.constant(1.0, shape=[2, 10])
        comm0 = hb.distribute.Communicator.build(shared_name, devices)
        recv0 = comm0.allreduce(a * 0.75)
        loss0 = tf.reduce_mean(recv0) * 20.0
      with tf.device('/gpu:1'):
        b = tf.constant(2.0, shape=[2, 10])
        comm1 = hb.distribute.Communicator.build(shared_name, devices)
        recv1 = comm1.allreduce(b)
        loss1 = tf.reduce_mean(recv1) * 10.0
      loss = loss0 * loss1
      grad0, grad1 = tf.gradients(
        [loss], [a, b], [2.0], colocate_gradients_with_ops=True)
      with hb.train.monitored_session() as sess:
        g0, g1 = sess.run([grad0, grad1])
        np.testing.assert_allclose(
          g0,
          [[82.5] * 10] * 2,
          rtol=1e-6)
        np.testing.assert_allclose(
          g1,
          [[110.0] * 10] * 2,
          rtol=1e-6)

  def test_allreduce_grad_error(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    shared_name = 'comm'
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        a = tf.constant(1.0, shape=[2, 10])
        comm0 = hb.distribute.Communicator.build(shared_name, devices)
        recv0 = comm0.allreduce(a * 0.75, reduce_op=hb.distribute.ops.MAX)
        loss0 = tf.reduce_mean(recv0) * 20.0
      with tf.device('/gpu:1'):
        b = tf.constant(2.0, shape=[2, 10])
        comm1 = hb.distribute.Communicator.build(shared_name, devices)
        recv1 = comm1.allreduce(b, reduce_op=hb.distribute.ops.MAX)
        loss1 = tf.reduce_mean(recv1) * 10.0
      loss = loss0 * loss1
      with self.assertRaises(RuntimeError):
        tf.gradients([loss], [a, b], [2.0])


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

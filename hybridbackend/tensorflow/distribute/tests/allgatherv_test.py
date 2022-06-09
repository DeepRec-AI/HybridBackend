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

r'''Tests for Allgatherv.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import unittest

import numpy as np
import tensorflow as tf

import hybridbackend.common.test as hbtest
import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
class AllgathervTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['NCCL_LAUNCH_MODE'] = 'GROUP'
    os.environ['TF_CPP_VMODULE'] = 'nccl_allgatherv=1'

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def _test_allgatherv(self, devices, shapes):
    hb.context.options.update(comm_pubsub_device='')

    num_comms = len(shapes)
    num_devices = len(devices)
    all_inputs = []
    allgathervs = []

    # Generate different shapes for different ranks
    def _v_shape(shape, i):
      if not shape:
        return shape
      shape[0] *= (i + 1)
      return shape

    with tf.Graph().as_default():
      prev_comm_gathervs = None
      for i in xrange(num_comms):
        comm_inputs = []
        comm_gathervs = []
        shared_name = f'comm_{i}'
        for d in xrange(num_devices):
          with tf.device(devices[d]):
            comm_inputs.append(
              tf.get_variable(
                f'input{i}/part_{d}',
                initializer=tf.random_normal(
                  _v_shape(shapes[i], i), mean=100, stddev=80)))
            comm = hb.distribute.Communicator.build(shared_name, devices)
            with tf.control_dependencies(
                [prev_comm_gathervs[d]] if prev_comm_gathervs else None):
              comm_gathervs.append(comm.allgatherv(comm_inputs[d]))
        prev_comm_gathervs = comm_gathervs
        all_inputs.append(comm_inputs)
        allgathervs.append(comm_gathervs)
      baselines = [tf.concat(all_inputs[i], axis=0)
                   for i in xrange(num_comms)]

      with hb.train.monitored_session() as sess:
        expects = sess.run(baselines)
        actuals = sess.run(allgathervs)
      for i in xrange(num_comms):
        for d in xrange(num_devices):
          np.testing.assert_allclose(actuals[i][d], expects[i], atol=1e-4)

  def test_one_device_one_tensor_allgatherv(self):
    self._test_allgatherv(
      [f'/job:localhost/replica:0/task:0/device:GPU:{d}' for d in [0]],
      [[16, 4]])

  def test_two_devices_four_tensors_allgatherv(self):
    self._test_allgatherv(
      [f'/job:localhost/replica:0/task:0/device:GPU:{d}' for d in [0, 1]],
      [[17, 41, 37], [19, 121], [1], [16, 6]])

  def test_allgatherv_grad_none(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    shared_name = 'comm'
    with tf.Graph().as_default() as graph:
      with tf.device('/gpu:0'):
        a = tf.constant(1.0, shape=[2, 10])
        comm0 = hb.distribute.Communicator.build(shared_name, devices)
        recv0 = comm0.allgatherv(a)
        loss0 = tf.reduce_mean(recv0) * 20.0
      with tf.device('/gpu:1'):
        b = tf.constant(2.0, shape=[2, 10])
        comm1 = hb.distribute.Communicator.build(shared_name, devices)
        recv1 = comm1.allgatherv(b * 0.75)
        loss1 = tf.reduce_mean(recv1) * 10.0
      loss = loss0 * loss1
      grad0, grad1 = tf.gradients([loss], [a, b], [2.0])
      np.testing.assert_equal(grad0, None)
      np.testing.assert_equal(grad1, None)
    graph.finalize()


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

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

r'''Tests for Allgather.
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
class AllgatherTest(unittest.TestCase):
  def _test_allgather(self, devices, shapes):
    hb.context.options.update(comm_pubsub_device='')

    num_comms = len(shapes)
    num_devices = len(devices)
    all_inputs = []
    allgathers = []
    allgathervs = []

    with tf.Graph().as_default():
      prev_comm_gathers = None
      prev_commv_gathers = None
      for i in xrange(num_comms):
        comm_inputs = []
        comm_gathers = []
        commv_gathers = []
        shared_name = f'comm_{i}'
        shared_namev = f'commv_{i}'
        for d in xrange(num_devices):
          with tf.device(devices[d]):
            comm_inputs.append(
              tf.get_variable(
                f'input{i}/part_{d}',
                initializer=tf.random_normal(
                  shapes[i], mean=100, stddev=80)))
            comm = hb.distribute.Communicator.build(shared_name, devices)
            with tf.control_dependencies(
                [prev_comm_gathers[d]] if prev_comm_gathers else None):
              ga1 = comm.allgather(comm_inputs[d])
            comm_gathers.append(ga1)
            commv = hb.distribute.Communicator.build(shared_namev, devices)
            with tf.control_dependencies(
                [prev_commv_gathers[d]] if prev_commv_gathers else None):
              ga2 = commv.allgatherv(comm_inputs[d])
            commv_gathers.append(ga2)
        prev_comm_gathers = comm_gathers
        prev_commv_gathers = commv_gathers
        all_inputs.append(comm_inputs)
        allgathers.append(comm_gathers)
        allgathervs.append(commv_gathers)
      baselines = [
        tf.concat(all_inputs[i], axis=0)
        if shapes[i] else tf.stack(all_inputs[i])
        for i in xrange(num_comms)]

      with hb.train.monitored_session() as sess:
        expects = sess.run(baselines)
        actuals = sess.run(allgathers)
        actuals_by_v = sess.run(allgathervs)
      for i in xrange(num_comms):
        for d in xrange(num_devices):
          np.testing.assert_allclose(actuals[i][d], expects[i], atol=1e-4)
          np.testing.assert_allclose(actuals_by_v[i][d], expects[i], atol=1e-4)

  def test_one_device_one_tensor_allgather(self):
    self._test_allgather(
      [f'/job:localhost/replica:0/task:0/device:GPU:{d}' for d in [0]],
      [[16, 4]])

  def test_two_devices_scalar_tensors_allgather(self):
    self._test_allgather(
      [f'/job:localhost/replica:0/task:0/device:GPU:{d}' for d in [0, 1]],
      [[]])

  def test_two_devices_four_tensors_allgather(self):
    self._test_allgather(
      [f'/job:localhost/replica:0/task:0/device:GPU:{d}' for d in [0, 1]],
      [[17, 41, 37], [19, 121], [1], [16, 6]])

  def test_allgather_grad(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    shared_name = 'comm'
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        a = tf.constant(1.0, shape=[2, 10])
        comm0 = hb.distribute.Communicator.build(shared_name, devices)
        recv0 = comm0.allgather(a)
        loss0 = tf.reduce_mean(recv0) * 20.0
      with tf.device('/gpu:1'):
        b = tf.constant(2.0, shape=[2, 10])
        comm1 = hb.distribute.Communicator.build(shared_name, devices)
        recv1 = comm1.allgather(b * 0.75)
        loss1 = tf.reduce_mean(recv1) * 10.0
      loss = loss0 * loss1
      grad0, grad1 = tf.gradients([loss], [a, b], [2.0])
      with hb.train.monitored_session() as sess:
        g0, g1 = sess.run([grad0, grad1])
        np.testing.assert_allclose(
          g0,
          [[25.0] * 10] * 2,
          rtol=1e-6)
        np.testing.assert_allclose(
          g1,
          [[18.75] * 10] * 2,
          rtol=1e-6)


if __name__ == '__main__':
  hbtest.register(['gpu', 'dist'])
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  os.environ['TF_CPP_VMODULE'] = 'nccl_allgather=1'
  unittest.main()

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

r'''Tests for Alltoall.
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
class AlltoallTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['TF_CPP_VMODULE'] = 'nccl_alltoall=1'

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_alltoall(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoall_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    shape = [2, 4]
    inputs = []
    expected = []
    actual = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(comm_id, devices)
            initializer = tf.random_normal(shape, mean=100, stddev=80)
            inputs.append(tf.get_variable(
              f'alltoall_input_{i}',
              initializer=initializer))
            transposed = comm.alltoall(inputs[i])
            actual.append(transposed)
        with server.monitored_session() as sess:
          results = sess.run({'actual': actual, 'inputs': inputs})
          expected = list(map(list, zip(*results['inputs'])))
          np.testing.assert_allclose(expected, results['actual'], rtol=1e-6)
    finally:
      del server

  def test_alltoall_grad(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoall_grad_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    values = [9.6, 8.8]
    shape = [2, 10]
    grad_ys = [2.0]
    all_ys = []
    all_outputs = []
    all_inputs = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(comm_id, devices)
            inputs = tf.constant(values[i], shape=shape)
            outputs = comm.alltoall(inputs)
            all_inputs.append(inputs)
            all_outputs.append(outputs)
            all_ys.append(tf.reduce_sum(outputs, axis=0))

        with tf.device(devices[0]):
          xs = all_inputs
          ys = [tf.reduce_sum(tf.add_n(all_ys))]
          actual = tf.gradients(
            ys, xs, grad_ys, colocate_gradients_with_ops=True)

          all_inputs_v2 = [tf.split(x, 2) for x in all_inputs]
          baseline_all_outputs = list(map(list, zip(*all_inputs_v2)))
          baseline_all_ys = [
            tf.reduce_sum(baseline_all_outputs, axis=0)]
          baseline_ys = [tf.reduce_sum(tf.add_n(baseline_all_ys))]
          expected = tf.gradients(
            baseline_ys, xs, grad_ys, colocate_gradients_with_ops=True)
        with server.monitored_session() as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          np.testing.assert_allclose(
            results['expected'], results['actual'], rtol=1e-6)
    finally:
      del server


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

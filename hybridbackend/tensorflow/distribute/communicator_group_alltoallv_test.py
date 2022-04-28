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

r'''Tests for GroupAlltoallv.
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
class GroupAlltoallvTest(unittest.TestCase):
  def test_group_alltoallv(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'group_alltoallv_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    try:
      with tf.Graph().as_default():
        train_ops = []
        with tf.device(devices[0]):
          comm0 = hb.distribute.Communicator.build(comm_id, devices)
          ids0 = tf.constant([1, 2, 3], dtype=tf.int64)
          sizes0 = tf.constant([1, 2], dtype=tf.int32)
          ids2 = tf.constant([7, 8, 9], dtype=tf.int64)
          sizes2 = tf.constant([2, 1], dtype=tf.int32)
          out02, out_sizes02 = comm0.group_alltoallv(
            [ids0, ids2], [sizes0, sizes2])
          train_ops.append([out02[0], out_sizes02[0], out02[1], out_sizes02[1]])
        with tf.device(devices[1]):
          comm1 = hb.distribute.Communicator.build(comm_id, devices)
          ids1 = tf.constant([4, 5, 6], dtype=tf.int64)
          sizes1 = tf.constant([2, 1], dtype=tf.int32)
          ids3 = tf.constant([3, 9, 6], dtype=tf.int64)
          sizes3 = tf.constant([1, 2], dtype=tf.int32)
          out13, out_sizes13 = comm1.group_alltoallv(
            [ids1, ids3], [sizes1, sizes3])
          train_ops.append([out13[0], out_sizes13[0], out13[1], out_sizes13[1]])
        with server.monitored_session() as sess:
          d0result, d1result = sess.run(train_ops)
          np.testing.assert_equal(d0result[0], [1, 4, 5])
          np.testing.assert_equal(d0result[1], [1, 2])
          np.testing.assert_equal(d0result[2], [7, 8, 3])
          np.testing.assert_equal(d0result[3], [2, 1])
          np.testing.assert_equal(d1result[0], [2, 3, 6])
          np.testing.assert_equal(d1result[1], [2, 1])
          np.testing.assert_equal(d1result[2], [9, 9, 6])
          np.testing.assert_equal(d1result[3], [1, 2])
    finally:
      del server

  def test_group_alltoallv_grad(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'group_alltoallv_grad_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    values = [[1.0, 2.4], [9.6, 8.8]]
    shapes = [[[6], [8]], [[10], [12]]]
    lrs = [1.0, 3.0]
    grad_ys = [2.0]
    all_ys = []
    all_inputs = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(comm_id, devices)
            inputs = [
              tf.constant(values[i][j], shape=shapes[i][j])
              for j, _ in enumerate(devices)]
            inputs = tf.concat(inputs, axis=0)
            all_inputs.append(inputs)
            sizes = tf.constant(
              [shapes[i][j][0] for j, _ in enumerate(devices)])
            outputs, _ = comm.group_alltoallv(
              [inputs], [sizes])
            all_ys.append(lrs[i] * tf.reduce_sum(outputs[0]))
        with tf.device(devices[0]):
          xs = all_inputs
          ys = [tf.add_n(all_ys)]
          actual = tf.gradients(
            ys, xs, grad_ys, colocate_gradients_with_ops=True)
        with server.monitored_session() as sess:
          actual = sess.run(actual)
          np.testing.assert_allclose(
            actual[0],
            [2., 2., 2., 2., 2., 2.,
             6., 6., 6., 6., 6., 6., 6., 6.],
            rtol=1e-6)
          np.testing.assert_allclose(
            actual[1],
            [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
             6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
            rtol=1e-6)
    finally:
      del server

  def test_group_alltoallv_grad_half_prec(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'group_alltoallv_grad_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    values = [[1.0, 2.4], [9.6, 8.8]]
    shapes = [[[6], [8]], [[10], [12]]]
    lrs = [1.0, 3.0]
    grad_ys = [2.0]
    all_ys = []
    all_inputs = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(comm_id, devices)
            inputs = [
              tf.constant(values[i][j], shape=shapes[i][j])
              for j, _ in enumerate(devices)]
            inputs = tf.concat(inputs, axis=0)
            all_inputs.append(inputs)
            sizes = tf.constant(
              [shapes[i][j][0] for j, _ in enumerate(devices)])
            outputs, _ = comm.group_alltoallv(
              [inputs], [sizes],
              wire_dtype=tf.float16)
            all_ys.append(lrs[i] * tf.reduce_sum(outputs[0]))
        with tf.device(devices[0]):
          xs = all_inputs
          ys = [tf.add_n(all_ys)]
          actual = tf.gradients(
            ys, xs, grad_ys, colocate_gradients_with_ops=True)
        with server.monitored_session() as sess:
          sess.run(actual)
    finally:
      del server


if __name__ == '__main__':
  hbtest.register(['gpu', 'dist'])
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  os.environ['TF_CPP_VMODULE'] = 'nccl_alltoallv=1,nccl_group_alltoallv=1'
  unittest.main()

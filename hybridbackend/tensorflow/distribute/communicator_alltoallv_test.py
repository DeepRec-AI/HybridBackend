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

r'''Tests for Alltoallv.
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
class AlltoallvTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['TF_CPP_VMODULE'] = 'nccl_alltoallv=1,nccl_group_alltoallv=1'

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_alltoallv(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallv_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    try:
      with tf.Graph().as_default():
        train_ops = []
        with tf.device(devices[0]):
          comm0 = hb.distribute.Communicator.build(comm_id, devices)
          ids0 = tf.constant([1, 2, 3], dtype=tf.int64)
          sizes0 = tf.constant([1, 2], dtype=tf.int32)
          out0, out_sizes0 = comm0.alltoallv(ids0, sizes0)
          train_ops.append([out0, out_sizes0])
        with tf.device(devices[1]):
          comm1 = hb.distribute.Communicator.build(comm_id, devices)
          ids1 = tf.constant([4, 5, 6], dtype=tf.int64)
          sizes1 = tf.constant([1, 2], dtype=tf.int32)
          out1, out_sizes1 = comm1.alltoallv(ids1, sizes1)
          train_ops.append([out1, out_sizes1])
        with server.monitored_session() as sess:
          d0result, d1result = sess.run(train_ops)
          np.testing.assert_allclose(d0result[0], [1, 4], rtol=1e-6)
          np.testing.assert_allclose(d0result[1], [1, 1], rtol=1e-6)
          np.testing.assert_allclose(d1result[0], [2, 3, 5, 6], rtol=1e-6)
          np.testing.assert_allclose(d1result[1], [2, 2], rtol=1e-6)
    finally:
      del server

  def test_alltoallv_grad(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallv_grad_test'
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
            outputs, _ = comm.alltoallv(inputs, sizes)
            all_ys.append(lrs[i] * tf.reduce_sum(outputs))
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

  def test_alltoallv_half_prec(self):
    hb.context.options.update(comm_pubsub_device='')

    os.environ['HB_COMM_WIRE_DTYPE_FLOAT'] = 'float16'
    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallv_half_prec_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    try:
      with tf.Graph().as_default():
        train_ops = []
        with tf.device(devices[0]):
          comm0 = hb.distribute.Communicator.build(comm_id, devices)
          ids0 = tf.constant(
            [1.0123, 2.3333, 3.4444], dtype=tf.float32)
          sizes0 = tf.constant([1, 2], dtype=tf.int32)
          out0, out_sizes0 = comm0.alltoallv(ids0, sizes0)
          train_ops.append([out0, out_sizes0])
        with tf.device(devices[1]):
          comm1 = hb.distribute.Communicator.build(comm_id, devices)
          ids1 = tf.constant(
            [4.3333, 5.4444, 6.5555], dtype=tf.float32)
          sizes1 = tf.constant([1, 2], dtype=tf.int32)
          out1, out_sizes1 = comm1.alltoallv(ids1, sizes1)
          train_ops.append([out1, out_sizes1])
        with server.monitored_session() as sess:
          d0result, d1result = sess.run(train_ops)
          np.testing.assert_allclose(d0result[0], [1.0123, 4.3333], rtol=1e-2)
          np.testing.assert_allclose(d0result[1], [1, 1], rtol=1e-6)
          np.testing.assert_allclose(
            d1result[0], [2.3333, 3.4444, 5.4444, 6.5555], rtol=1e-2)
          np.testing.assert_allclose(d1result[1], [2, 2], rtol=1e-6)
    finally:
      del server
    del os.environ['HB_COMM_WIRE_DTYPE_FLOAT']

  def test_alltoallv_multi_steps(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallv_multistep_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    shapes = [[10230 + 5260, 64], [4000 + 990, 64]]
    sizes = [[10230, 5260], [4000, 990]]
    train_ops = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(comm_id, devices)
            merged = tf.get_variable(
              f'input_{i}',
              initializer=tf.random_normal(
                shapes[i], mean=100, stddev=80))
            out, out_sizes = comm.alltoallv(
              merged, sizes[i], common_shape=[64])
            train_ops.append(tf.group([out, out_sizes]))
        with server.monitored_session() as sess:
          for _ in xrange(100):
            sess.run(train_ops)
    finally:
      del server

  def test_multi_alltoallv(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'multi_alltoallv_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    try:
      with tf.Graph().as_default():
        train_ops = []
        with tf.device(devices[0]):
          comm0 = hb.distribute.Communicator.build(comm_id, devices)
          ids0 = tf.constant([1, 2, 3], dtype=tf.int64)
          sizes0 = tf.constant([1, 2], dtype=tf.int32)
          out0, out_sizes0 = comm0.alltoallv(ids0, sizes0)
          ids2 = tf.constant([7, 8, 9], dtype=tf.int64)
          sizes2 = tf.constant([2, 1], dtype=tf.int32)
          with tf.control_dependencies([out0, out_sizes0]):
            out2, out_sizes2 = comm0.alltoallv(ids2, sizes2)
          train_ops.append([out0, out_sizes0, out2, out_sizes2])
        with tf.device(devices[1]):
          comm1 = hb.distribute.Communicator.build(comm_id, devices)
          ids1 = tf.constant([4, 5, 6], dtype=tf.int64)
          sizes1 = tf.constant([2, 1], dtype=tf.int32)
          out1, out_sizes1 = comm1.alltoallv(ids1, sizes1)
          ids3 = tf.constant([3, 9, 6], dtype=tf.int64)
          sizes3 = tf.constant([1, 2], dtype=tf.int32)
          with tf.control_dependencies([out1, out_sizes1]):
            out3, out_sizes3 = comm1.alltoallv(ids3, sizes3)
          train_ops.append([out1, out_sizes1, out3, out_sizes3])
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

  def test_mutli_alltoallv_multi_steps(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'multi_alltoallv_multistep_test'
    num_comms = 2
    server = hb.train.Server({'localhost': ['localhost:0']})
    train_ops = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          prev_op = None
          with tf.device(d):
            for c in xrange(num_comms):
              shapes = [[1023 + c + 526 + c, 64], [400 + c + 99 + c, 64]]
              sizes = [[1023 + c, 526 + c], [400 + c, 99 + c]]
              merged = tf.get_variable(
                f'comm_{c}/input_{i}',
                initializer=tf.random_normal(
                  shapes[i], mean=100, stddev=80))
              comm = hb.distribute.Communicator.build(
                f'{comm_id}_{c}', devices)
              with tf.control_dependencies(
                  None if prev_op is None else [prev_op]):
                out, out_sizes = comm.alltoallv(
                  merged, sizes[i], common_shape=[64])
              train_op = tf.group([out, out_sizes])
              prev_op = train_op
              train_ops.append(train_op)
        with server.monitored_session() as sess:
          for _ in xrange(100):
            sess.run(train_ops)
    finally:
      del server

  def test_reuse_comm_mutli_alltoallv_multi_steps(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'reuse_comm_mutli_alltoallv_multi_steps_test'
    num_ops = 2
    server = hb.train.Server({'localhost': ['localhost:0']})
    train_ops = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(comm_id, devices)
            for c in xrange(num_ops):
              shapes = [[1023 + c + 526 + c, 64], [400 + c + 99 + c, 64]]
              sizes = [[1023 + c, 526 + c], [400 + c, 99 + c]]
              merged = tf.get_variable(
                f'comm_{c}/input_{i}',
                initializer=tf.random_normal(
                  shapes[i], mean=100, stddev=80))
              if c == 0:
                out, out_sizes = comm.alltoallv(
                  merged, sizes[i], common_shape=[64])
                train_ops.append(tf.group([out, out_sizes]))
              else:
                with tf.control_dependencies([train_ops[-1]]):
                  out, out_sizes = comm.alltoallv(
                    merged, sizes[i], common_shape=[64])
                  train_ops.append(tf.group([out, out_sizes]))
        with server.monitored_session() as sess:
          for _ in xrange(100):
            sess.run(train_ops)
    finally:
      del server


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

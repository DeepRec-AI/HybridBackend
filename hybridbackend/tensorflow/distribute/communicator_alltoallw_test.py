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

r'''Tests for Alltoallw.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import unittest

import numpy as np
import tensorflow as tf

import hybridbackend.tensorflow as hb
import hybridbackend.test as hbtest


# pylint: disable=missing-docstring
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
class AlltoallwTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['TF_CPP_VMODULE'] = 'nccl_alltoallw=1'

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_alltoallw(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallw_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    shapes = [[[1023, 4, 16], [526, 4, 16]], [[400, 4, 16], [99, 4, 16]]]
    expected = []
    actual = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(comm_id, devices)
            inputs = [
              tf.get_variable(
                f'input_{i}/from_{j}',
                initializer=tf.random_normal(
                  shapes[i][j], mean=100, stddev=80))
              for j, _ in enumerate(devices)]
            transposed = comm.alltoallw(inputs, common_shape=[4, 16])
            actual.append(transposed)
            expected.append(inputs)
        expected = list(map(list, zip(*expected)))  # transpose the inputs.
        with server.monitored_session() as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          for act, exp in zip(results['actual'], results['expected']):
            for acti, expi in zip(act, exp):
              np.testing.assert_allclose(expi, acti, rtol=1e-6)
    finally:
      del server

  def test_alltoallw_fp16(self):
    hb.context.options.update(comm_pubsub_device='')

    os.environ['HB_COMM_WIRE_DTYPE_FLOAT'] = 'float16'
    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallw_fp16_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    shapes = [[[1023, 4, 16], [526, 4, 16]], [[400, 4, 16], [99, 4, 16]]]
    expected = []
    actual = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(
              comm_id, devices, impl=hb.distribute.NcclCommunicator)
            inputs = [
              tf.get_variable(
                f'input_{i}/from_{j}',
                initializer=tf.random_normal(
                  shapes[i][j], mean=100, stddev=80))
              for j, _ in enumerate(devices)]
            transposed = comm.alltoallw(inputs, common_shape=[4, 16])
            actual.append(transposed)
            expected.append(inputs)
        expected = list(map(list, zip(*expected)))  # transpose the inputs.
        with server.monitored_session() as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          for act, exp in zip(results['actual'], results['expected']):
            for acti, expi in zip(act, exp):
              np.testing.assert_allclose(expi, acti, rtol=1e-2)
    finally:
      del server
    del os.environ['HB_COMM_WIRE_DTYPE_FLOAT']

  def test_alltoallw_grad(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallw_grad_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    values = [[1.0, 2.4], [9.6, 8.8]]
    shapes = [[[6, 2, 3], [8, 2, 3]], [[10, 2, 3], [12, 2, 3]]]
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
            all_inputs.append(inputs)
            outputs = comm.alltoallw(inputs, common_shape=[2, 3])
            all_ys.append(
              tf.reduce_sum(tf.concat(outputs, axis=0)) * lrs[i])
        with tf.device(devices[0]):
          xs = [y for x in all_inputs for y in x]
          ys = [tf.add_n(all_ys)]
          actual = tf.gradients(
            ys, xs, grad_ys, colocate_gradients_with_ops=True)
          baseline_all_outputs = list(map(list, zip(*all_inputs)))
          baseline_all_ys = [
            tf.reduce_sum(
              tf.concat(baseline_all_outputs[i], axis=0)) * lrs[i]
            for i, _ in enumerate(devices)]
          baseline_ys = [tf.add_n(baseline_all_ys)]
          expected = tf.gradients(
            baseline_ys, xs, grad_ys, colocate_gradients_with_ops=True)
        with server.monitored_session() as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          for act, exp in zip(results['actual'], results['expected']):
            for acti, expi in zip(act, exp):
              np.testing.assert_allclose(expi, acti, rtol=1e-6)
    finally:
      del server

  def test_alltoallw_grad_fp16(self):
    hb.context.options.update(comm_pubsub_device='')

    os.environ['HB_COMM_WIRE_DTYPE_FLOAT'] = 'float16'
    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallw_grad_fp16_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    values = [[1.0, 2.4], [9.6, 8.8]]
    shapes = [[[6, 2, 3], [8, 2, 3]], [[10, 2, 3], [12, 2, 3]]]
    lrs = [1.0, 3.0]
    grad_ys = [2.0]
    all_ys = []
    all_inputs = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(
              comm_id, devices, impl=hb.distribute.NcclCommunicator)
            inputs = [
              tf.constant(values[i][j], shape=shapes[i][j])
              for j, _ in enumerate(devices)]
            all_inputs.append(inputs)
            outputs = comm.alltoallw(inputs, common_shape=[2, 3])
            all_ys.append(
              tf.reduce_sum(tf.concat(outputs, axis=0)) * lrs[i])
        with tf.device(devices[0]):
          xs = [y for x in all_inputs for y in x]
          ys = [tf.add_n(all_ys)]
          actual = tf.gradients(
            ys, xs, grad_ys, colocate_gradients_with_ops=True)
          baseline_all_outputs = list(map(list, zip(*all_inputs)))
          baseline_all_ys = [
            tf.reduce_sum(
              tf.concat(baseline_all_outputs[i], axis=0)) * lrs[i]
            for i, _ in enumerate(devices)]
          baseline_ys = [tf.add_n(baseline_all_ys)]
          expected = tf.gradients(
            baseline_ys, xs, grad_ys, colocate_gradients_with_ops=True)
        with server.monitored_session() as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          for act, exp in zip(results['actual'], results['expected']):
            for acti, expi in zip(act, exp):
              np.testing.assert_allclose(expi, acti, rtol=1e-2)
    finally:
      del server
    del os.environ['HB_COMM_WIRE_DTYPE_FLOAT']

  def test_alltoallw_multi_steps(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallw_multistep_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    shapes = [[[10230, 64], [5260, 64]], [[4000, 64], [990, 64]]]
    train_ops = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(comm_id, devices)
            inputs = [
              tf.get_variable(
                f'input_{i}/from_{j}',
                initializer=tf.random_normal(
                  shapes[i][j], mean=100, stddev=80))
              for j, _ in enumerate(devices)]
            transposed = comm.alltoallw(inputs, common_shape=[64])
            train_op = [
              tf.matmul(m, m, transpose_a=True) for m in transposed]
            train_ops.append(train_op)
        with server.monitored_session() as sess:
          for _ in xrange(100):
            sess.run(train_ops)
    finally:
      del server

  def test_mutli_alltoallw_multi_steps(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'multi_alltoallw_multistep_test'
    num_comms = 2
    server = hb.train.Server({'localhost': ['localhost:0']})
    train_ops = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            for c in xrange(num_comms):
              shapes = [
                [[1023 + c, 64], [526 + c, 64]],
                [[400 + c, 64], [99 + c, 64]]]
              inputs = [
                tf.get_variable(
                  f'comm_{c}/input_{i}/from_{j}',
                  initializer=tf.random_normal(
                    shapes[i][j], mean=100, stddev=80))
                for j, _ in enumerate(devices)]
              comm = hb.distribute.Communicator.build(
                f'{comm_id}_{c}', devices)
              transposed = comm.alltoallw(inputs, common_shape=[64])
              train_op = [
                tf.matmul(m, m, transpose_a=True) for m in transposed]
              train_ops.append(train_op)
        with server.monitored_session() as sess:
          for _ in xrange(100):
            sess.run(train_ops)
    finally:
      del server


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

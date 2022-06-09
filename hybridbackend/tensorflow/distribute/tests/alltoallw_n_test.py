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

r'''Tests for coalesced Alltoallw.
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
class AlltoallwTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['NCCL_LAUNCH_MODE'] = 'GROUP'
    os.environ['TF_CPP_VMODULE'] = 'nccl_alltoallw=1'

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_alltoallw_n(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallw_n_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    shapes_list = [
      [[[1023, 4, 3], [526, 4, 3]], [[400, 4, 3], [99, 4, 3]]],
      [[[34, 2, 3], [22, 2, 3]], [[11, 2, 3], [44, 2, 3]]],
      [[[520, 5], [1314, 5]], [[888, 5], [6, 5]]]]
    num_columns = len(shapes_list)
    expected = []
    actual = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(
              comm_id, devices, impl=hb.distribute.NcclCommunicator)
            inputs_list = []
            for k in xrange(num_columns):
              inputs_list.append([
                tf.get_variable(
                  f'from_{i}/to_{j}/var_{k}',
                  initializer=tf.random_normal(
                    shapes_list[k][i][j], mean=100, stddev=80))
                for j, _ in enumerate(devices)])
            transposed = comm.alltoallw_n(
              inputs_list, common_shapes=[[4, 3], [2, 3], [5]])
            actual.append(transposed)
            expected.append(inputs_list)
        expected = zip(*expected)
        expected = list(zip(*[map(list, zip(*p)) for p in expected]))
        with server.monitored_session() as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          for act, exp in zip(results['actual'], results['expected']):
            for acti, expi in zip(act, exp):
              for actj, expj in zip(acti, expi):
                np.testing.assert_allclose(expj, actj, rtol=1e-6)
    finally:
      del server

  def test_alltoallw_n_fp16(self):
    hb.context.options.update(comm_pubsub_device='')

    os.environ['HB_COMM_WIRE_DTYPE_FLOAT'] = 'float16'
    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallw_n_fp16_test'
    server = hb.train.Server({'localhost': ['localhost:0']})
    shapes_list = [
      [[[1023, 4, 3], [526, 4, 3]], [[400, 4, 3], [99, 4, 3]]],
      [[[34, 2, 3], [22, 2, 3]], [[11, 2, 3], [44, 2, 3]]],
      [[[520, 5], [1314, 5]], [[888, 5], [6, 5]]]]
    num_columns = len(shapes_list)
    expected = []
    actual = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = hb.distribute.Communicator.build(
              comm_id, devices, impl=hb.distribute.NcclCommunicator)
            inputs_list = []
            for k in xrange(num_columns):
              inputs_list.append([
                tf.get_variable(
                  f'from_{i}/to_{j}/var_{k}',
                  initializer=tf.random_normal(
                    shapes_list[k][i][j], mean=100, stddev=80))
                for j, _ in enumerate(devices)])
            transposed = comm.alltoallw_n(
              inputs_list, common_shapes=[[4, 3], [2, 3], [5]])
            actual.append(transposed)
            expected.append(inputs_list)
        expected = zip(*expected)
        expected = list(zip(*[map(list, zip(*p)) for p in expected]))
        with server.monitored_session() as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          for act, exp in zip(results['actual'], results['expected']):
            for acti, expi in zip(act, exp):
              for actj, expj in zip(acti, expi):
                np.testing.assert_allclose(expj, actj, rtol=1e-2)
    finally:
      del server
    del os.environ['HB_COMM_WIRE_DTYPE_FLOAT']

  def test_alltoallw_n_grad(self):
    hb.context.options.update(comm_pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallw_n_grad_test'
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
            outputs = comm.alltoallw_n([inputs], common_shapes=[[2, 3]])
            all_ys.append(
              lrs[i] * tf.reduce_sum(
                tf.concat(outputs[0], axis=0)))
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
              for actj, expj in zip(acti, expi):
                np.testing.assert_allclose(expj, actj, rtol=1e-6)
    finally:
      del server

  def test_alltoallw_n_grad_fp16(self):
    hb.context.options.update(comm_pubsub_device='')

    os.environ['HB_COMM_WIRE_DTYPE_FLOAT'] = 'float16'
    devices = ['/gpu:0', '/gpu:1']
    comm_id = 'alltoallw_n_grad_fp16_test'
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
            outputs = comm.alltoallw_n([inputs], common_shapes=[[2, 3]])
            all_ys.append(
              lrs[i] * tf.reduce_sum(
                tf.concat(outputs[0], axis=0)))
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
              for actj, expj in zip(acti, expi):
                np.testing.assert_allclose(expj, actj, rtol=1e-2)
    finally:
      del server
    del os.environ['HB_COMM_WIRE_DTYPE_FLOAT']


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

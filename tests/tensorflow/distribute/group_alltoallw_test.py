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

import numpy as np
import os
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
import unittest

from hybridbackend.tensorflow.distribute.communicator import Communicator
from hybridbackend.tensorflow.distribute.nccl import NcclCommunicator
from hybridbackend.tensorflow.framework.context import context
from hybridbackend.tensorflow.training.server import MonitoredTrainingSession
from hybridbackend.tensorflow.training.server import Server

from tests.tensorflow.spawn import register


# pylint: disable=missing-docstring
class AlltoallwTest(unittest.TestCase):
  def test_group_alltoallw(self):
    context.update_params(pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = "group_alltoallw_test"
    server = Server({"localhost": ["localhost:0"]})
    group_shapes = [
        [[[1023, 4, 3], [526, 4, 3]], [[400, 4, 3], [99, 4, 3]]],
        [[[34, 2, 3], [22, 2, 3]], [[11, 2, 3], [44, 2, 3]]],
        [[[520, 5], [1314, 5]], [[888, 5], [6, 5]]]]
    group_size = len(group_shapes)
    expected = []
    actual = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = Communicator.create(comm_id, devices, impl=NcclCommunicator)
            group_inputs = []
            for k in xrange(group_size):
              group_inputs.append([
                  tf.get_variable(
                      f'from_{i}/to_{j}/var_{k}',
                      initializer=tf.random_normal(
                          group_shapes[k][i][j], mean=100, stddev=80))
                  for j, _ in enumerate(devices)])
            transposed = comm.group_alltoallw(
                group_inputs, common_shapes=[[4, 3], [2, 3], [5]])
            actual.append(transposed)
            expected.append(group_inputs)
        expected = zip(*expected)
        expected = list(zip(*[map(list, zip(*p)) for p in expected]))
        with MonitoredTrainingSession(server.target) as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          for act, exp in zip(results['actual'], results['expected']):
            for acti, expi in zip(act, exp):
              for actj, expj in zip(acti, expi):
                np.testing.assert_allclose(expj, actj, rtol=1e-6)
    finally:
      del server

  def test_group_alltoallw_fp16(self):
    context.update_params(pubsub_device='')

    os.environ['HB_COMM_WIRE_DTYPE_FLOAT'] = 'float16'
    devices = ['/gpu:0', '/gpu:1']
    comm_id = "group_alltoallw_fp16_test"
    server = Server({"localhost": ["localhost:0"]})
    group_shapes = [
        [[[1023, 4, 3], [526, 4, 3]], [[400, 4, 3], [99, 4, 3]]],
        [[[34, 2, 3], [22, 2, 3]], [[11, 2, 3], [44, 2, 3]]],
        [[[520, 5], [1314, 5]], [[888, 5], [6, 5]]]]
    group_size = len(group_shapes)
    expected = []
    actual = []
    try:
      with tf.Graph().as_default():
        for i, d in enumerate(devices):
          with tf.device(d):
            comm = Communicator.create(comm_id, devices, impl=NcclCommunicator)
            group_inputs = []
            for k in xrange(group_size):
              group_inputs.append([
                  tf.get_variable(
                      f'from_{i}/to_{j}/var_{k}',
                      initializer=tf.random_normal(
                          group_shapes[k][i][j], mean=100, stddev=80))
                  for j, _ in enumerate(devices)])
            transposed = comm.group_alltoallw(
                group_inputs, common_shapes=[[4, 3], [2, 3], [5]])
            actual.append(transposed)
            expected.append(group_inputs)
        expected = zip(*expected)
        expected = list(zip(*[map(list, zip(*p)) for p in expected]))
        with MonitoredTrainingSession(server.target) as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          for act, exp in zip(results['actual'], results['expected']):
            for acti, expi in zip(act, exp):
              for actj, expj in zip(acti, expi):
                np.testing.assert_allclose(expj, actj, rtol=1e-2)
    finally:
      del server
    del os.environ['HB_COMM_WIRE_DTYPE_FLOAT']

  def test_group_alltoallw_grad(self):
    context.update_params(pubsub_device='')

    devices = ['/gpu:0', '/gpu:1']
    comm_id = "group_alltoallw_grad_test"
    server = Server({"localhost": ["localhost:0"]})
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
            comm = Communicator.create(comm_id, devices, impl=NcclCommunicator)
            inputs = [
                tf.constant(values[i][j], shape=shapes[i][j])
                for j, _ in enumerate(devices)]
            all_inputs.append(inputs)
            outputs = comm.group_alltoallw([inputs], common_shapes=[[2, 3]])
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
        with MonitoredTrainingSession(server.target) as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          for act, exp in zip(results['actual'], results['expected']):
            for acti, expi in zip(act, exp):
              for actj, expj in zip(acti, expi):
                np.testing.assert_allclose(expj, actj, rtol=1e-6)
    finally:
      del server

  def test_group_alltoallw_grad_fp16(self):
    context.update_params(pubsub_device='')

    os.environ['HB_COMM_WIRE_DTYPE_FLOAT'] = 'float16'
    devices = ['/gpu:0', '/gpu:1']
    comm_id = "group_alltoallw_grad_fp16_test"
    server = Server({"localhost": ["localhost:0"]})
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
            comm = Communicator.create(comm_id, devices, impl=NcclCommunicator)
            inputs = [
                tf.constant(values[i][j], shape=shapes[i][j])
                for j, _ in enumerate(devices)]
            all_inputs.append(inputs)
            outputs = comm.group_alltoallw([inputs], common_shapes=[[2, 3]])
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
        with MonitoredTrainingSession(server.target) as sess:
          results = sess.run({'actual': actual, 'expected': expected})
          for act, exp in zip(results['actual'], results['expected']):
            for acti, expi in zip(act, exp):
              for actj, expj in zip(acti, expi):
                np.testing.assert_allclose(expj, actj, rtol=1e-2)
    finally:
      del server
    del os.environ['HB_COMM_WIRE_DTYPE_FLOAT']


if __name__ == '__main__':
  register(['gpu', 'dist'])
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  os.environ['TF_CPP_VMODULE'] = 'nccl_alltoallw=1'
  unittest.main()

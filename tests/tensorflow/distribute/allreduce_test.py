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

r'''Tests for Allreduce.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from six.moves import xrange # pylint: disable=redefined-builtin
import unittest

from tests.tensorflow.spawn import Spawn
from tests.tensorflow.spawn import register


# pylint: disable=missing-docstring
def _test_simple_allreduce(rank, a, b):
  # pylint: disable=import-outside-toplevel
  import tensorflow as tf

  from hybridbackend.tensorflow.distribute.communicator import Communicator
  from hybridbackend.tensorflow.framework.context import context
  from hybridbackend.tensorflow.training.server_lib import device_setter
  from hybridbackend.tensorflow.training.server import MonitoredTrainingSession
  from hybridbackend.tensorflow.training.server import Server

  with tf.Graph().as_default():
    server = Server()
    with tf.device(device_setter()):
      comm0 = Communicator.create('comm0', context.devices)
      input0 = tf.constant(a) if rank == 0 else tf.constant(b)
      sum0 = comm0.allreduce(input0)
    with MonitoredTrainingSession(server.target, is_chief=True) as sess:
      return sess.run(sum0)


def _test_simple_allreduce_max(rank, a, b):
  # pylint: disable=import-outside-toplevel
  import tensorflow as tf

  from hybridbackend.tensorflow.distribute.communicator import Communicator
  from hybridbackend.tensorflow.framework.context import context
  from hybridbackend.tensorflow.framework.ops import CollectiveOps
  from hybridbackend.tensorflow.training.server_lib import device_setter
  from hybridbackend.tensorflow.training.server import MonitoredTrainingSession
  from hybridbackend.tensorflow.training.server import Server

  with tf.Graph().as_default():
    server = Server()
    with tf.device(device_setter()):
      comm0 = Communicator.create('comm0', context.devices)
      input0 = tf.constant(a) if rank == 0 else tf.constant(b)
      sum0 = comm0.allreduce(input0, reduce_op=CollectiveOps.MAX)
    with MonitoredTrainingSession(server.target, is_chief=True) as sess:
      return sess.run(sum0)


def _test_simple_allreduce_multicomm(rank, a, b, ncomms):
  # pylint: disable=import-outside-toplevel
  import tensorflow as tf

  from hybridbackend.tensorflow.distribute.communicator import Communicator
  from hybridbackend.tensorflow.framework.context import context
  from hybridbackend.tensorflow.training.server_lib import device_setter
  from hybridbackend.tensorflow.training.server import MonitoredTrainingSession
  from hybridbackend.tensorflow.training.server import Server

  with tf.Graph().as_default():
    server = Server()
    with tf.device(device_setter()):
      sums = []
      for icomm in xrange(ncomms):
        commi = Communicator.create(f'comm{icomm}', context.devices)
        inputi = tf.constant(a) if rank == 0 else tf.constant(b + icomm)
        sums.append(commi.allreduce(inputi))
    with MonitoredTrainingSession(server.target, is_chief=True) as sess:
      return sess.run(sums)


class AllreduceTest(unittest.TestCase):
  def test_simple_allreduce(self):
    a = 13
    b = 22
    results = Spawn(2)(lambda rank: _test_simple_allreduce(rank, a, b))
    np.testing.assert_allclose(results[0], a + b, rtol=1e-6)
    np.testing.assert_allclose(results[1], a + b, rtol=1e-6)

  def test_onedevice_allreduce(self):
    a = 13
    results = Spawn(1)(lambda rank: _test_simple_allreduce(rank, a, 22))
    np.testing.assert_allclose(results[0], a, rtol=1e-6)

  def test_allreduce_max(self):
    a = 13
    b = 22
    results = Spawn(2)(lambda rank: _test_simple_allreduce_max(rank, a, b))
    np.testing.assert_allclose(results[0], b, rtol=1e-6)
    np.testing.assert_allclose(results[1], b, rtol=1e-6)

  def test_multicomm(self):
    a = 13
    b = 22
    ncomms = 10
    results = Spawn(2)(
        lambda rank: _test_simple_allreduce_multicomm(rank, a, b, ncomms))
    for icomm in xrange(ncomms):
      np.testing.assert_allclose(results[0][icomm], a + b + icomm, rtol=1e-6)
      np.testing.assert_allclose(results[1][icomm], a + b + icomm, rtol=1e-6)


if __name__ == '__main__':
  register(['gpu', 'dist'])
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  os.environ['TF_CPP_VMODULE'] = 'nccl_allreduce=1'
  unittest.main()

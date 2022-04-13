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

r'''Tests for Broadcast.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import unittest

from tests.tensorflow.spawn import Spawn
from tests.tensorflow.spawn import register


def _test_broadcast(rank, a, b):
  r'''Test Broadcast.
  '''
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
      comm = Communicator.create('comm0', context.devices)
      root = tf.constant(a) if rank == 0 else tf.constant(b)
      recv = comm.broadcast(root, root_rank=0)
    with MonitoredTrainingSession(server.target, is_chief=True) as sess:
      return sess.run(recv)


# pylint: disable=missing-docstring
class BroadcastTest(unittest.TestCase):
  def test_broadcast(self):
    a = 13
    b = 22
    results = Spawn(2)(lambda rank: _test_broadcast(rank, a, b))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6)
    np.testing.assert_allclose(results[0], a, rtol=1e-6)


if __name__ == '__main__':
  register(['gpu', 'dist'])
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  os.environ['TF_CPP_VMODULE'] = 'nccl_broadcast=1'
  unittest.main()

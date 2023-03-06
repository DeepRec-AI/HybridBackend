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

r'''Tests for broadcast collective communication.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np

import hybridbackend.common.test as hbtest

# pylint: disable=missing-docstring,import-outside-toplevel


def _test_broadcast(rank, a, b):
  r'''Test Broadcast.
  '''
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default():
    with hb.scope():
      data = tf.constant(a) if rank == 0 else tf.constant(b)
      recv = hb.distribute.broadcast(data, root_rank=0)
      with tf.train.MonitoredTrainingSession('') as sess:
        return sess.run(recv)


@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_NCCL') == 'ON', 'NCCL required')
class BroadcastTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['TF_CPP_VMODULE'] = (
      'nccl_comm=1,'
      'nccl_create=1,'
      'nccl_broadcast=1')

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_broadcast(self):
    a = 13
    b = 22
    results = hbtest.Spawn(2)(lambda rank: _test_broadcast(rank, a, b))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6)
    np.testing.assert_allclose(results[0], a, rtol=1e-6)


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

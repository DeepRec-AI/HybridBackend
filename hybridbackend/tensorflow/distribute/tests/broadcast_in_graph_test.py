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

r'''Tests for Broadcast in one graph.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np
import tensorflow as tf

import hybridbackend.common.test as hbtest
import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_NCCL') == 'ON', 'NCCL required')
class BroadcastInGraphTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['NCCL_LAUNCH_MODE'] = 'GROUP'
    os.environ['TF_CPP_VMODULE'] = 'nccl_broadcast=1'

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_broadcast_in_graph(self):
    hb.context.options.update(comm_pubsub_device='')

    a = 13
    b = 22
    devices = ['/gpu:0', '/gpu:1']
    shared_name = 'comm'
    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        comm0 = hb.distribute.Communicator.build(shared_name, devices)
        root = tf.constant(a)
        recv0 = comm0.broadcast(root, root_rank=0)
      with tf.device('/gpu:1'):
        comm1 = hb.distribute.Communicator.build(shared_name, devices)
        noop = tf.constant(b)
        recv1 = comm1.broadcast(noop, root_rank=0)
      with hb.train.monitored_session() as sess:
        s0, s1 = sess.run([recv0, recv1])
        np.testing.assert_allclose(s0, a, rtol=1e-6)
        np.testing.assert_allclose(s1, a, rtol=1e-6)


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

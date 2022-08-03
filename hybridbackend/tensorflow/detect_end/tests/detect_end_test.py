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

r'''Test for out-of-range detect.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np

import hybridbackend.common.test as hbtest


# pylint: disable=missing-docstring
def _test_single(_):
  r'''Testing on a single worker
  '''
  # pylint: disable=import-outside-toplevel
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  batch_size = 10

  with tf.Graph().as_default():
    with hb.scope(mode=tf.estimator.ModeKeys.TRAIN):
      with tf.device('/cpu:0'):
        ds = tf.data.Dataset.range(100)
        ds = ds.batch(batch_size=batch_size)
        iterator = hb.data.make_one_shot_iterator(ds)
        batch = iterator.get_next()
      with hb.train.monitored_session() as sess:
        final_result = None
        while not sess.should_stop():
          final_result = sess.run(batch)
        return final_result


def _test_distributed(rank):
  r'''Testing on multiple distributed workers
  '''
  # pylint: disable=import-outside-toplevel
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  batch_size = 10

  with tf.Graph().as_default():
    with hb.scope(mode=tf.estimator.ModeKeys.TRAIN):
      with tf.device('/cpu:0'):
        ds = tf.data.Dataset.range(100 + rank * 50)
        ds = ds.batch(batch_size=batch_size)
        iterator = hb.data.make_one_shot_iterator(ds)
        batch = iterator.get_next()
      with hb.train.monitored_session() as sess:
        final_result = None
        while not sess.should_stop():
          final_result = sess.run(batch)
        return final_result


@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_NCCL') == 'ON', 'NCCL required')
class DetectEndTest(unittest.TestCase):
  r'''Tests for the out-of-range sync.
  '''
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

  def test_single(self):
    results = hbtest.Spawn()(_test_single)
    np.testing.assert_equal(
      results[0], [90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

  def test_parallel(self):
    results = hbtest.Spawn(2)(_test_distributed)
    np.testing.assert_equal(results[0], results[1])


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

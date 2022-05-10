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

r'''Round robin shuffle Op Test.
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
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
class FloorModShuffleTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

  def test_cpu(self):
    np.random.seed(0)
    x = np.random.randint(
      low=-1000000000,
      high=1000000000,
      size=10000,
      dtype=np.int32)
    num_partitions = 5

    with tf.Graph().as_default():
      with tf.device('/cpu:0'):
        xi = tf.constant(x)
        y_list, idx_list = hb.math.floormod_partition(xi, num_partitions)
        y, ysizes, idx = hb.math.floormod_shuffle(xi, num_partitions)
        with hb.train.monitored_session() as sess:
          tf_y_list, tf_idx_list, tf_y, tf_ysizes, tf_idx = sess.run(
            [y_list, idx_list, y, ysizes, idx])

    np.testing.assert_equal(len(tf_y), len(tf_idx))
    np.testing.assert_equal(len(tf_ysizes), num_partitions)
    np.testing.assert_equal(x, np.take(tf_y, tf_idx))
    np.testing.assert_equal(len(tf_y_list), len(tf_idx_list))
    np.testing.assert_equal(
      np.concatenate(tf_y_list, axis=0),
      np.take(
        x,
        np.concatenate(tf_idx_list, axis=0)))
    np.testing.assert_equal(
      np.concatenate(tf_y_list, axis=0),
      tf_y)

  def test_gpu(self):
    np.random.seed(0)
    x = np.random.randint(
      low=-1000000000,
      high=1000000000,
      size=100000,
      dtype=np.int64)
    num_partitions = 3

    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        xi = tf.constant(x)
        y, ysizes, idx = hb.math.floormod_shuffle(xi, num_partitions)
        with hb.train.monitored_session() as sess:
          tf_y, tf_ysizes, tf_idx = sess.run([y, ysizes, idx])

    np.testing.assert_equal(len(tf_y), len(tf_idx))
    np.testing.assert_equal(len(tf_ysizes), num_partitions)
    np.testing.assert_equal(x, np.take(tf_y, tf_idx))

  def test_empty_input(self):
    x = np.array([], np.int64)
    num_partitions = 7

    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        xi = tf.constant(x)
        y, ysizes, idx = hb.math.floormod_shuffle(xi, num_partitions)
      with hb.train.monitored_session() as sess:
        tf_y, tf_ysizes, tf_idx = sess.run([y, ysizes, idx])

    np.testing.assert_equal(len(tf_y), len(tf_idx))
    np.testing.assert_equal(len(tf_ysizes), num_partitions)
    for i in range(num_partitions):
      np.testing.assert_equal(0, tf_ysizes[i])

  def test_gpu_group(self):
    num_columns = 10
    num_partitions = 3
    np.random.seed(0)
    x = [
      np.random.randint(
        low=-1000000000,
        high=1000000000,
        size=100000,
        dtype=np.int64)
      for i in range(num_columns)]

    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        xt = [tf.constant(t) for t in x]
        y, ysizes, idx = hb.math.group_floormod_shuffle(xt, num_partitions)
        with hb.train.monitored_session() as sess:
          result = sess.run({'y': y, 'ysizes': ysizes, 'idx': idx})

    for c in range(num_columns):
      np.testing.assert_equal(len(result['y'][c]), len(result['idx'][c]))
      np.testing.assert_equal(len(result['ysizes'][c]), num_partitions)
      np.testing.assert_equal(x[c], np.take(result['y'][c], result['idx'][c]))

  def test_empty_input_group(self):
    num_columns = 3
    num_partitions = 7
    x = [np.array([], np.int64) for i in range(num_columns)]

    with tf.Graph().as_default():
      with tf.device('/gpu:0'):
        xt = [tf.constant(t) for t in x]
        y, ysizes, idx = hb.math.group_floormod_shuffle(xt, num_partitions)
      with hb.train.monitored_session() as sess:
        result = sess.run({'y': y, 'ysizes': ysizes, 'idx': idx})

    for c in range(num_columns):
      np.testing.assert_equal(len(result['y'][c]), len(result['idx'][c]))
      np.testing.assert_equal(len(result['ysizes'][c]), num_partitions)
      for i in range(num_partitions):
        np.testing.assert_equal(0, result['ysizes'][c][i])


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

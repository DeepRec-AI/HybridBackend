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

r'''Tests for alltoall collective communication.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np

import hybridbackend.common.test as hbtest

# pylint: disable=missing-docstring,import-outside-toplevel


def _test_alltoall(rank, world_size, shape):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default(), hb.scope():
    inputs = []
    for d in range(world_size):
      inputs.append(
        tf.get_variable(
          f'alltoall_input/replica{d}',
          dtype=tf.float32,
          initializer=tf.random_normal(shape, mean=100, stddev=80, seed=d)))
    actual = hb.distribute.alltoall(inputs[rank])

    with tf.train.MonitoredTrainingSession('') as sess:
      results = sess.run({'actual': actual, 'inputs': inputs})
      results['actual'] = [o.tolist() for o in results['actual']]
      results['inputs'] = [o.tolist() for o in results['inputs']]
      results['expected'] = list(map(list, zip(*results['inputs'])))
      return results


def _test_alltoall_grad(rank, world_size, shape, g):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default(), hb.scope():
    inputs = []
    for d in range(world_size):
      inputs.append(
        tf.get_variable(
          f'alltoall_input/replica{d}',
          dtype=tf.float32,
          initializer=tf.random_normal(shape, mean=100, stddev=80, seed=d)))
    shuffled = hb.distribute.alltoall(inputs[rank])
    loss = tf.reduce_mean(shuffled)
    grad = tf.gradients(
      [loss], [inputs[rank]], [g],
      colocate_gradients_with_ops=True)

    with tf.train.MonitoredTrainingSession('') as sess:
      return sess.run(grad)


def _test_alltoallv(rank, ids_list, sizes_list, dtype=None, wire_dtype=None):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default(), hb.scope(comm_wire_dtype=wire_dtype):
    ids = tf.constant(
      ids_list[rank], dtype=tf.int64 if dtype is None else dtype)
    sizes = tf.constant(sizes_list[rank], dtype=tf.int32)
    out_ids, out_sizes = hb.distribute.alltoall(ids, sizes=sizes)

    with tf.train.MonitoredTrainingSession('') as sess:
      return sess.run({'ids': out_ids, 'sizes': out_sizes})


def _test_alltoallv_fp16(rank, ids_list, sizes_list):
  import tensorflow as tf

  return _test_alltoallv(
    rank, ids_list, sizes_list, dtype=tf.float32, wire_dtype=tf.float16)


def _test_alltoallv_grad(rank, values, sizes, g):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default(), hb.scope():
    inputs = tf.constant(
      values[rank],
      dtype=tf.float32,
      shape=(sum(sizes[rank]),))
    input_sizes = tf.constant(
      sizes[rank],
      dtype=tf.int32)
    outputs, _ = hb.distribute.alltoall(inputs, sizes=input_sizes)
    loss = tf.reduce_mean(outputs)
    grad = tf.gradients(
      [loss], [inputs], [g],
      colocate_gradients_with_ops=True)

    with tf.train.MonitoredTrainingSession('') as sess:
      return sess.run(grad)


def _test_alltoallv_n(rank, ids_and_sizes_list, wire_dtype=None):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default(), hb.scope(comm_wire_dtype=wire_dtype):
    out_ids_list = []
    out_sizes_list = []
    for ids_and_sizes in ids_and_sizes_list[rank]:
      ids = tf.constant(ids_and_sizes['ids'], dtype=tf.float32)
      sizes = tf.constant(ids_and_sizes['sizes'], dtype=tf.int32)
      out_ids, out_sizes = hb.distribute.alltoall(ids, sizes=sizes)
      out_ids_list.append(out_ids)
      out_sizes_list.append(out_sizes)

    with tf.train.MonitoredTrainingSession('') as sess:
      results = sess.run({'ids': out_ids_list, 'sizes': out_sizes_list})
      return [
        {'ids': ids.tolist(), 'sizes': results['sizes'][i].tolist()}
        for i, ids in enumerate(results['ids'])]


def _test_alltoallv_n_fp16(rank, ids_and_sizes_list):
  import tensorflow as tf

  return _test_alltoallv_n(rank, ids_and_sizes_list, wire_dtype=tf.float16)


def _test_alltoallv_n_grad(rank, ids_and_sizes_list, g):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default(), hb.scope():
    grads = []
    for ids_and_sizes in ids_and_sizes_list[rank]:
      inputs = tf.constant(
        ids_and_sizes['ids'],
        dtype=tf.float32,
        shape=(sum(ids_and_sizes['sizes']),))
      input_sizes = tf.constant(
        ids_and_sizes['sizes'],
        dtype=tf.int32)
      outputs, _ = hb.distribute.alltoall(inputs, sizes=input_sizes)
      loss = tf.reduce_mean(outputs)
      grads.append(
        tf.gradients(
          [loss], [inputs], [g],
          colocate_gradients_with_ops=True))

    with tf.train.MonitoredTrainingSession('') as sess:
      results = sess.run(grads)
      return [r[0].tolist() for r in results]


@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_NCCL') == 'ON', 'NCCL required')
class AlltoallTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['TF_CPP_VMODULE'] = (
      'nccl_comm=1,'
      'nccl_create=1,'
      'nccl_alltoall=1,'
      'nccl_alltoallv=1,'
      'packing=2,'
      'optimize_collective=2')

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def xtest_alltoall(self):
    world_size = 2
    results = hbtest.Spawn(world_size)(
      lambda rank: _test_alltoall(rank, world_size, [2, 3]))
    for d in range(world_size):
      np.testing.assert_allclose(
        results[d]['actual'], results[d]['expected'][d], rtol=1e-6)

  def xtest_alltoall_grad(self):
    world_size = 2
    w = 2
    h = 10
    g = 2.0
    results = hbtest.Spawn(world_size)(
      lambda rank: _test_alltoall_grad(rank, world_size, [w, h], g))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6)
    np.testing.assert_allclose(
      results[0], [[[g / (w * h)] * h] * w], rtol=1e-6)

  def xtest_alltoallv(self):
    results = hbtest.Spawn(2)(
      lambda rank: _test_alltoallv(
        rank, [[1, 2, 3], [4, 5, 6]], [[1, 2], [1, 2]]))
    np.testing.assert_allclose(results[0]['ids'], [1, 4], rtol=1e-6)
    np.testing.assert_allclose(results[0]['sizes'], [1, 1], rtol=1e-6)
    np.testing.assert_allclose(results[1]['ids'], [2, 3, 5, 6], rtol=1e-6)
    np.testing.assert_allclose(results[1]['sizes'], [2, 2], rtol=1e-6)

  def xtest_alltoallv_grad(self):
    g = 2.0
    sizes = [[5, 1], [3, 4]]
    results = hbtest.Spawn(2)(
      lambda rank: _test_alltoallv_grad(
        rank, [3.6, 4.2], sizes, g))
    g0 = g / (sizes[0][0] + sizes[1][0])
    g1 = g / (sizes[0][1] + sizes[1][1])
    np.testing.assert_allclose(
      results[0],
      [sizes[0][0] * [g0] + sizes[0][1] * [g1]],
      rtol=1e-6)
    np.testing.assert_allclose(
      results[1],
      [sizes[1][0] * [g0] + sizes[1][1] * [g1]],
      rtol=1e-6)

  def xtest_alltoallv_fp16(self):
    results = hbtest.Spawn(2)(
      lambda rank: _test_alltoallv_fp16(
        rank, [[1., 2., 3.], [4., 5., 6.]], [[1, 2], [1, 2]]))
    np.testing.assert_allclose(results[0]['ids'], [1., 4.], rtol=1e-6)
    np.testing.assert_allclose(results[0]['sizes'], [1, 1], rtol=1e-6)
    np.testing.assert_allclose(results[1]['ids'], [2., 3., 5., 6.], rtol=1e-6)
    np.testing.assert_allclose(results[1]['sizes'], [2, 2], rtol=1e-6)

  def test_alltoallv_n(self):
    results = hbtest.Spawn(2)(
      lambda rank: _test_alltoallv_n(
        rank,
        {0: [{'ids': [1., 2., 3.], 'sizes': [1, 2]},
             {'ids': [4., 5., 6.], 'sizes': [2, 1]}],
         1: [{'ids': [7., 8., 9.], 'sizes': [2, 1]},
             {'ids': [10., 11., 12.], 'sizes': [1, 2]}]}))
    np.testing.assert_allclose(results[0][0]['ids'], [1., 7., 8.], rtol=1e-6)
    np.testing.assert_equal(results[0][0]['sizes'], [1, 2])
    np.testing.assert_allclose(results[1][0]['ids'], [2., 3., 9.], rtol=1e-6)
    np.testing.assert_equal(results[1][0]['sizes'], [2, 1])
    np.testing.assert_allclose(results[0][1]['ids'], [4., 5., 10.], rtol=1e-6)
    np.testing.assert_equal(results[0][1]['sizes'], [2, 1])
    np.testing.assert_allclose(results[1][1]['ids'], [6., 11., 12.], rtol=1e-6)
    np.testing.assert_equal(results[1][1]['sizes'], [1, 2])

  def xtest_alltoallv_n_fp16(self):
    results = hbtest.Spawn(2)(
      lambda rank: _test_alltoallv_n_fp16(
        rank,
        {0: [{'ids': [1., 2., 3.], 'sizes': [1, 2]},
             {'ids': [4., 5., 6.], 'sizes': [2, 1]}],
         1: [{'ids': [7., 8., 9.], 'sizes': [2, 1]},
             {'ids': [10., 11., 12.], 'sizes': [1, 2]}]}))
    np.testing.assert_allclose(results[0][0]['ids'], [1., 7., 8.], rtol=1e-6)
    np.testing.assert_equal(results[0][0]['sizes'], [1, 2])
    np.testing.assert_allclose(results[1][0]['ids'], [2., 3., 9.], rtol=1e-6)
    np.testing.assert_equal(results[1][0]['sizes'], [2, 1])
    np.testing.assert_allclose(results[0][1]['ids'], [4., 5., 10.], rtol=1e-6)
    np.testing.assert_equal(results[0][1]['sizes'], [2, 1])
    np.testing.assert_allclose(results[1][1]['ids'], [6., 11., 12.], rtol=1e-6)
    np.testing.assert_equal(results[1][1]['sizes'], [1, 2])

  def xtest_alltoallv_n_grad(self):
    results = hbtest.Spawn(2)(
      lambda rank: _test_alltoallv_n_grad(
        rank,
        {0: [{'ids': [1., 2., 3.], 'sizes': [1, 2]},
             {'ids': [4., 5., 6.], 'sizes': [2, 1]}],
         1: [{'ids': [7., 8., 9.], 'sizes': [2, 1]},
             {'ids': [10., 11., 12.], 'sizes': [1, 2]}]},
        2.0))
    np.testing.assert_allclose(
      results[0][0], [0.666667, 0.666667, 0.666667], rtol=1e-6)
    np.testing.assert_allclose(
      results[0][1], [0.666667, 0.666667, 0.666667], rtol=1e-6)
    np.testing.assert_allclose(
      results[1][0], [0.666667, 0.666667, 0.666667], rtol=1e-6)
    np.testing.assert_allclose(
      results[1][1], [0.666667, 0.666667, 0.666667], rtol=1e-6)


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

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

r'''Tests for allgather collective communication.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np

import hybridbackend.common.test as hbtest

# pylint: disable=missing-docstring,import-outside-toplevel


# Generate different shapes for different ranks
def _v_shape(shape, i):
  if not shape:
    return shape
  shape[0] *= (i + 1)
  return shape


def _test_allgather(rank, world_size, shapes):
  r'''Test Allgather.
  '''
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  all_inputs = []
  allgathers = []
  allgathervs = []
  with tf.Graph().as_default(), hb.scope():
    for i, shape in enumerate(shapes):
      comm_inputs = []
      for d in range(world_size):
        comm_inputs.append(
          tf.get_variable(
            f'input{i}/replicas{d}',
            initializer=tf.random_normal(
              shape,
              mean=100,
              stddev=80,
              seed=i * len(shapes) * 100 + d)))
      all_inputs.append(comm_inputs)
      allgathers.append(
        hb.distribute.allgather(comm_inputs[rank], varying_size=False))
      allgathervs.append(hb.distribute.allgather(comm_inputs[rank]))
    baselines = [
      tf.concat(all_inputs[i], 0) if shape else tf.stack(all_inputs[i])
      for i, shape in enumerate(shapes)]
    with tf.train.MonitoredTrainingSession('') as sess:
      results = sess.run({
        'baselines': baselines,
        'allgathers': allgathers,
        'allgathervs': allgathervs})
      results['baselines'] = [o.tolist() for o in results['baselines']]
      results['allgathers'] = [o.tolist() for o in results['allgathers']]
      results['allgathervs'] = [o.tolist() for o in results['allgathervs']]
      return results


def _test_allgatherv(rank, world_size, shapes):
  r'''Test Allgatherv.
  '''
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  all_inputs = []
  allgathervs = []
  with tf.Graph().as_default(), hb.scope():
    for i, shape in enumerate(shapes):
      comm_inputs = []
      for d in range(world_size):
        comm_inputs.append(
          tf.get_variable(
            f'input{i}/replicas{d}',
            initializer=tf.random_normal(
              _v_shape(shape, i),
              mean=100,
              stddev=80,
              seed=i * len(shapes) * 100 + d)))
      all_inputs.append(comm_inputs)
      allgathervs.append(hb.distribute.allgather(comm_inputs[rank]))
    baselines = [
      tf.concat(all_inputs[i], 0) if shape else tf.stack(all_inputs[i])
      for i, shape in enumerate(shapes)]
    with tf.train.MonitoredTrainingSession('') as sess:
      results = sess.run({
        'baselines': baselines,
        'allgathervs': allgathervs})
      results['baselines'] = [o.tolist() for o in results['baselines']]
      results['allgathervs'] = [o.tolist() for o in results['allgathervs']]
      return results


@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON', 'GPU required')
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_NCCL') == 'ON', 'NCCL required')
class AllgatherTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['TF_CPP_VMODULE'] = (
      'nccl_comm=1,'
      'nccl_create=1,'
      'nccl_allgather=1,'
      'nccl_allgatherv=1')

  def tearDown(self):  # pylint: disable=invalid-name
    del os.environ['TF_CPP_VMODULE']
    del os.environ['CUDA_VISIBLE_DEVICES']

  def _test_allgather(self, world_size, shapes):
    results = hbtest.Spawn(world_size)(
      lambda rank: _test_allgather(rank, world_size, shapes))
    for d in range(world_size):
      r = results[d]
      for i, _ in enumerate(shapes):
        np.testing.assert_allclose(
          r['baselines'][i], r['allgathers'][i], atol=1e-4)
        np.testing.assert_allclose(
          r['baselines'][i], r['allgathervs'][i], atol=1e-4)

  def test_allgather_scalar(self):
    self._test_allgather(2, [[]])

  def test_allgather_fallback(self):
    self._test_allgather(1, [[6, 14]])

  def test_allgather_n(self):
    self._test_allgather(2, [[17, 3, 4], [2, 3], [1], [16, 6]])

  def _test_allgatherv(self, world_size, shapes):
    results = hbtest.Spawn(world_size)(
      lambda rank: _test_allgatherv(rank, world_size, shapes))
    for d in range(world_size):
      r = results[d]
      for i, _ in enumerate(shapes):
        np.testing.assert_allclose(
          r['baselines'][i], r['allgathervs'][i], atol=1e-4)

  def test_allgatherv_scalar(self):
    self._test_allgatherv(2, [[]])

  def test_allgatherv_fallback(self):
    self._test_allgatherv(1, [[6, 14]])

  def test_allgatherv_n(self):
    self._test_allgatherv(2, [[2, 3, 4], [2, 3], [1], [2, 6]])


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

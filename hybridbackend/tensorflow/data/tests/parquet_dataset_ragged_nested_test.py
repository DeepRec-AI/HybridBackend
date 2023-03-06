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

r'''Parquet batch dataset nested ragged tensors test.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf

import hybridbackend.common.test as hbtest
import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
class ParquetDatasetRaggedNestedTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    self._workspace = tempfile.mkdtemp()
    self._filename = os.path.join(
      self._workspace, 'ragged_test_pyarrow.parquet')
    self._data = pa.array(
      [[[1], [2, 3]], [[4], [5]]], pa.list_(pa.list_(pa.int64())))
    table = pa.Table.from_arrays([self._data], ['A'])
    pq.write_table(table, self._filename, compression='ZSTD')

  def tearDown(self):  # pylint: disable=invalid-name
    os.remove(self._filename)
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_read(self):
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(
        [self._filename],
        batch_size=2)
      ds = ds.apply(hb.data.rebatch(2))
      ds = ds.prefetch(4)
      batch = tf.data.make_one_shot_iterator(ds).get_next()

    with tf.Session(graph=graph) as sess:
      actual = sess.run(batch)['A'].to_list()
      expected = self._data.to_pylist()
      np.testing.assert_equal(actual, expected)

  def test_apply_to_sparse(self):
    with tf.Graph().as_default() as graph:
      ds = hb.data.Dataset.from_parquet([self._filename])
      ds = ds.batch(2)
      batch = tf.data.make_one_shot_iterator(ds).get_next()['A']
      baseline = tf.ragged.constant(self._data.to_pylist()).to_sparse()

    with tf.Session(graph=graph) as sess:
      actual, expected = sess.run([batch, baseline])
      np.testing.assert_equal(actual.indices, expected.indices)
      np.testing.assert_equal(actual.values, expected.values)
      np.testing.assert_equal(actual.dense_shape, expected.dense_shape)

  def test_apply_to_tensor(self):
    with tf.Graph().as_default() as graph:
      ds = hb.data.Dataset.from_parquet([self._filename], sparse_to_dense=True)
      ds = ds.batch(2)
      batch = tf.data.make_one_shot_iterator(ds).get_next()['A']
      baseline = tf.ragged.constant(self._data.to_pylist()).to_tensor()

    with tf.Session(graph=graph) as sess:
      actual, expected = sess.run([batch, baseline])
      np.testing.assert_equal(actual, expected)


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

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

r'''Parquet batch dataset rebatching test.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tempfile
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

import hybridbackend.common.test as hbtest
import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
class ParquetDatasetRebatchTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    self._workspace = tempfile.mkdtemp()
    self._filename = os.path.join(self._workspace, 'test.parquet')
    nrows = 100
    cols = list('ABCDE')

    def build_row(br, er, bseq, eseq):
      a = np.random.randint(br, er, dtype=np.int64)
      b = np.random.randint(
        br, er, size=(np.random.randint(bseq, eseq),), dtype=np.int64)
      c = np.random.randint(br, er, dtype=np.int64)
      d = np.random.randint(br, er, size=(4,), dtype=np.int64)
      e = np.array(list(map(
        str,
        np.random.randint(
          br, er * 100,
          size=(np.random.randint(bseq, eseq),),
          dtype=np.int64))))
      return [a, b, c, d, e]
    self._df = pd.DataFrame(
      np.array([build_row(0, 100, 1, 5) for _ in xrange(nrows)], dtype=object),
      columns=cols)
    self._df.to_parquet(self._filename)

  def tearDown(self):  # pylint: disable=invalid-name
    os.remove(self._filename)
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_pasthrough(self):
    batch_size = 20
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(self._filename, batch_size)
      ds = ds.apply(hb.data.rebatch(batch_size))
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    a = self._df['A']
    c = self._df['C']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        np.testing.assert_equal(result['A'], a[start_row:end_row].to_numpy())
        np.testing.assert_equal(result['C'], c[start_row:end_row].to_numpy())

  def test_expilict_batch(self):
    micro_batch_size = 20
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(self._filename, micro_batch_size)
      ds = ds.apply(hb.data.rebatch(batch_size))
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    a = self._df['A']
    c = self._df['C']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        np.testing.assert_equal(result['A'], a[start_row:end_row].to_numpy())
        np.testing.assert_equal(result['C'], c[start_row:end_row].to_numpy())

  def test_min_batch(self):
    micro_batch_size = 20
    batch_size = 32
    min_batch_size = 30
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(self._filename, micro_batch_size)
      ds = ds.apply(hb.data.rebatch(batch_size, min_batch_size))
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    a = self._df['A']
    c = self._df['C']
    with tf.Session(graph=graph) as sess:
      end_row = 0
      for _ in xrange(3):
        result = sess.run(batch)
        aresult = result['A']
        cresult = result['C']
        bs = len(aresult)
        start_row = end_row
        end_row = start_row + bs
        np.testing.assert_equal(aresult, a[start_row:end_row].to_numpy())
        np.testing.assert_equal(cresult, c[start_row:end_row].to_numpy())

  def test_ragged(self):
    micro_batch_size = 20
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(self._filename, micro_batch_size)
      ds = ds.apply(hb.data.rebatch(batch_size))
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    b = self._df['B']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected_items = b[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = hb.data.DataFrame.Value(
          np.array(expected_values),
          [np.array(expected_splits, dtype=np.int32)])
        actual = result['B']
        np.testing.assert_allclose(actual.values, expected.values)
        np.testing.assert_equal(
          actual.nested_row_splits, expected.nested_row_splits)

  def test_to_sparse(self):
    micro_batch_size = 20
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(self._filename, micro_batch_size)
      ds = ds.apply(hb.data.rebatch(batch_size))
      ds = ds.apply(hb.data.parse())
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    b = self._df['B']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected_items = b[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = hb.data.DataFrame.Value(
          np.array(expected_values),
          [np.array(expected_splits, dtype=np.int32)])
        actual = result['B']
        np.testing.assert_equal(actual.values, expected.values)
        np.testing.assert_equal(
          len(set(list(zip(*actual.indices))[0])) + 1,
          len(expected.nested_row_splits[0]))

  def test_shuffle_micro_batch(self):
    micro_batch_size = 20
    batch_size = 32
    with tf.Graph().as_default() as graph:
      srcds = hb.data.ParquetDataset(self._filename, micro_batch_size)
      ds = srcds.shuffle(4)
      ds = ds.apply(
        hb.data.rebatch(batch_size, fields=srcds.fields, num_parallel_scans=2))
      ds = ds.apply(hb.data.parse())
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    with tf.Session(graph=graph) as sess:
      for _ in xrange(3):
        sess.run(batch)

  def test_thread_pool(self):
    micro_batch_size = 20
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(self._filename, micro_batch_size)
      ds = ds.apply(hb.data.rebatch(batch_size, num_parallel_scans=3))
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    b = self._df['B']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected_items = b[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = hb.data.DataFrame.Value(
          np.array(expected_values),
          [np.array(expected_splits, dtype=np.int32)])
        actual = result['B']
        np.testing.assert_allclose(actual.values, expected.values)
        np.testing.assert_equal(
          actual.nested_row_splits, expected.nested_row_splits)

  def test_ragged_with_shape(self):
    micro_batch_size = 20
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(
        self._filename,
        micro_batch_size,
        fields=[hb.data.DataFrame.Field('D', shape=[4])])
      ds = ds.apply(hb.data.rebatch(batch_size))
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    d = self._df['D']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected_items = d[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = np.reshape(np.array(expected_values), (-1, 4))
        actual = result['D']
        np.testing.assert_allclose(actual.values, expected)


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

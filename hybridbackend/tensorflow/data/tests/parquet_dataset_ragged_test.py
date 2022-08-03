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

r'''Parquet batch dataset ragged tensors test.
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
class ParquetDatasetRaggedTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    self._workspace = tempfile.mkdtemp()
    self._filename = os.path.join(self._workspace, 'ragged_test.parquet')
    num_cols = 3
    self._df = pd.DataFrame(
      np.array([
        [
          np.random.randint(
            0, 100,
            size=(np.random.randint(1, 5),),
            dtype=np.int64)
          for _ in xrange(num_cols)]
        for _ in xrange(100)], dtype=object),
      columns=[f'col{c}' for c in xrange(num_cols)])
    self._df.to_parquet(self._filename)

  def tearDown(self):  # pylint: disable=invalid-name
    os.remove(self._filename)
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_read(self):
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(
        [self._filename],
        batch_size=batch_size,
        fields=['col2', 'col0'])
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    c = self._df['col0']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = hb.data.DataFrame.Value(
          np.array(expected_values),
          [np.array(expected_splits, dtype=np.int32)])
        actual = result['col0']
        np.testing.assert_allclose(actual.values, expected.values)
        np.testing.assert_equal(
          actual.nested_row_splits, expected.nested_row_splits)

  def test_to_sparse(self):
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(
        [self._filename],
        batch_size=batch_size,
        fields=['col2', 'col0'])
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()
      batch = hb.data.DataFrame.to_sparse(batch)

    c = self._df['col0']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = hb.data.DataFrame.Value(
          np.array(expected_values),
          [np.array(expected_splits, dtype=np.int32)])
        actual = result['col0']
        np.testing.assert_allclose(actual.values, expected.values)
        np.testing.assert_equal(
          len(set(list(zip(*actual.indices))[0])) + 1,
          len(expected.nested_row_splits[0]))

  def test_map_to_sparse(self):
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(
        [self._filename],
        batch_size=batch_size,
        fields=['col2', 'col0'])
      ds = ds.map(hb.data.DataFrame.to_sparse)
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    c = self._df['col0']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = hb.data.DataFrame.Value(
          np.array(expected_values),
          [np.array(expected_splits, dtype=np.int32)])
        actual = result['col0']
        np.testing.assert_allclose(actual.values, expected.values)
        np.testing.assert_equal(
          len(set(list(zip(*actual.indices))[0])) + 1,
          len(expected.nested_row_splits[0]))

  def test_apply_to_sparse(self):
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(
        [self._filename],
        batch_size=batch_size,
        fields=['col2', 'col0'])
      ds = ds.apply(hb.data.to_sparse())
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    c = self._df['col0']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = hb.data.DataFrame.Value(
          np.array(expected_values),
          [np.array(expected_splits, dtype=np.int32)])
        actual = result['col0']
        np.testing.assert_equal(actual.values, expected.values)
        np.testing.assert_equal(
          len(set(list(zip(*actual.indices))[0])) + 1,
          len(expected.nested_row_splits[0]))

  def test_feedable_iterator(self):
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(
        [self._filename],
        batch_size=batch_size,
        fields=['col2', 'col0'])
      ds = ds.apply(hb.data.to_sparse())
      ds = ds.prefetch(4)
      it = hb.data.make_one_shot_iterator(ds)
      handle_tensor = it.string_handle()
      feedable_handle = tf.placeholder(tf.string, shape=[])
      feedable_it = tf.data.Iterator.from_string_handle(
        feedable_handle,
        it.output_types,
        it.output_shapes,
        it.output_classes)
      batch = feedable_it.get_next()

    c = self._df['col0']
    with tf.Session(graph=graph) as sess:
      handle_val = sess.run(handle_tensor)
      for i in xrange(3):
        result = sess.run(batch, feed_dict={feedable_handle: handle_val})
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = hb.data.DataFrame.Value(
          np.array(expected_values),
          [np.array(expected_splits, dtype=np.int32)])
        actual = result['col0']
        np.testing.assert_equal(actual.values, expected.values)
        np.testing.assert_equal(
          len(set(list(zip(*actual.indices))[0])) + 1,
          len(expected.nested_row_splits[0]))

  def test_read_and_map(self):
    batch_size = 32
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(
        [self._filename],
        batch_size=batch_size,
        fields=['col2', 'col0'])

      def _parse(values):
        return values['col0'], values['col2']
      ds = ds.map(_parse)
      ds = ds.prefetch(4)
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    c = self._df['col0']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = hb.data.DataFrame.Value(
          np.array(expected_values),
          [np.array(expected_splits, dtype=np.int32)])
        actual = result[0]
        np.testing.assert_allclose(actual.values, expected.values)
        np.testing.assert_equal(
          actual.nested_row_splits, expected.nested_row_splits)


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

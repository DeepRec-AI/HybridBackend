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

import numpy as np
import pandas as pd
import os
import tempfile

from tensorflow.python.framework import ops
from tensorflow.python.platform import test

from hybridbackend.tensorflow.data import DataFrame
from hybridbackend.tensorflow.data import make_one_shot_iterator
from hybridbackend.tensorflow.data import ParquetDataset
from hybridbackend.tensorflow.data import rebatch
from hybridbackend.tensorflow.data import to_sparse


# pylint: disable=missing-docstring
class ParquetDatasetRebatchTest(test.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    self._workspace = tempfile.mkdtemp()
    self._filename = os.path.join(self._workspace, 'test.parquet')
    nrows = 100
    cols = list('ABCDE')
    def build_row(br, er, bseq, eseq):
      a = np.random.randint(br, er, dtype=np.int64)
      b = np.random.randint(
          br, er, size=(np.random.randint(bseq, eseq),), dtype=np.int64)
      c = np.random.randint(br, er, dtype=np.int64)
      d = np.random.randint(
          br, er, size=(np.random.randint(bseq, eseq),), dtype=np.int64)
      e = np.array(list(map(
          str,
          np.random.randint(
              br, er * 100,
              size=(np.random.randint(bseq, eseq),),
              dtype=np.int64))))
      return [a, b, c, d, e]
    self._df = pd.DataFrame(
        np.array([build_row(0, 100, 1, 5) for _ in range(nrows)]),
        columns=cols)
    self._df.to_parquet(self._filename)

  def tearDown(self):  # pylint: disable=invalid-name
    os.remove(self._filename)

  def test_pasthrough(self):
    batch_size = 20
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(self._filename, batch_size)
      ds = ds.apply(rebatch(batch_size))
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    a = self._df['A']
    c = self._df['C']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        self.assertAllEqual(result['A'], a[start_row:end_row].to_numpy())
        self.assertAllEqual(result['C'], c[start_row:end_row].to_numpy())

  def test_expilict_batch(self):
    micro_batch_size = 20
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(self._filename, micro_batch_size)
      ds = ds.apply(rebatch(batch_size))
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    a = self._df['A']
    c = self._df['C']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        self.assertAllEqual(result['A'], a[start_row:end_row].to_numpy())
        self.assertAllEqual(result['C'], c[start_row:end_row].to_numpy())

  def test_min_batch(self):
    micro_batch_size = 20
    batch_size = 32
    min_batch_size = 30
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(self._filename, micro_batch_size)
      ds = ds.apply(rebatch(batch_size, min_batch_size))
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    a = self._df['A']
    c = self._df['C']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      end_row = 0
      for _ in range(3):
        result = sess.run(batch)
        aresult = result['A']
        cresult = result['C']
        bs = len(aresult)
        start_row = end_row
        end_row = start_row + bs
        self.assertAllEqual(aresult, a[start_row:end_row].to_numpy())
        self.assertAllEqual(cresult, c[start_row:end_row].to_numpy())

  def test_ragged(self):
    micro_batch_size = 20
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(self._filename, micro_batch_size)
      ds = ds.apply(rebatch(batch_size))
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    b = self._df['B']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        expected_items = b[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = DataFrame.Value(
            np.array(expected_values),
            tuple([np.array(expected_splits, dtype=np.int32)]))
        actual = result['B']
        self.assertAllClose(actual, expected)

  def test_to_sparse(self):
    micro_batch_size = 20
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(self._filename, micro_batch_size)
      ds = ds.apply(rebatch(batch_size))
      ds = ds.apply(to_sparse())
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    b = self._df['B']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        expected_items = b[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = DataFrame.Value(
            np.array(expected_values),
            tuple([np.array(expected_splits, dtype=np.int32)]))
        actual = result['B']
        self.assertAllEqual(actual.values, expected.values)
        self.assertAllEqual(
            len(set(list(zip(*actual.indices))[0])) + 1,
            len(expected.nested_row_splits[0]))

  def test_shuffle_micro_batch(self):
    micro_batch_size = 20
    batch_size = 32
    with ops.Graph().as_default() as graph:
      srcds = ParquetDataset(self._filename, micro_batch_size)
      ds = srcds.shuffle(4)
      ds = ds.apply(
          rebatch(batch_size, fields=srcds.fields, num_parallel_scans=2))
      ds = ds.apply(to_sparse())
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    with self.test_session(use_gpu=False, graph=graph) as sess:
      for _ in range(3):
        sess.run(batch)

  def test_thread_pool(self):
    micro_batch_size = 20
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(self._filename, micro_batch_size)
      ds = ds.apply(rebatch(batch_size, num_parallel_scans=3))
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    b = self._df['B']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        expected_items = b[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = DataFrame.Value(
            np.array(expected_values),
            tuple([np.array(expected_splits, dtype=np.int32)]))
        actual = result['B']
        self.assertAllClose(actual, expected)


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  test.main()

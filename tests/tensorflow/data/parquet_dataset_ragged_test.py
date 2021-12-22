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

import numpy as np
import pandas as pd
import os
import tempfile

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from hybridbackend.tensorflow.data import DataFrame
from hybridbackend.tensorflow.data import make_one_shot_iterator
from hybridbackend.tensorflow.data import ParquetDataset
from hybridbackend.tensorflow.data import to_sparse


# pylint: disable=missing-docstring
class ParquetDatasetRaggedTest(test.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    self._workspace = tempfile.mkdtemp()
    self._filename = os.path.join(self._workspace, 'ragged_test.parquet')
    num_cols = 3
    self._df = pd.DataFrame(
        np.array(
            [
                [
                    np.random.randint(
                        0, 100,
                        size=(np.random.randint(1, 5),),
                        dtype=np.int64)
                    for _ in range(num_cols)]
                for _ in range(100)]),
        columns=[f'col{c}' for c in range(num_cols)])
    self._df.to_parquet(self._filename)

  def tearDown(self):  # pylint: disable=invalid-name
    os.remove(self._filename)

  def test_read(self):
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(
          [self._filename],
          batch_size=batch_size,
          fields=['col2', 'col0'])
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    c = self._df['col0']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = DataFrame.Value(
            np.array(expected_values),
            tuple([np.array(expected_splits, dtype=np.int32)]))
        actual = result['col0']
        self.assertAllClose(actual, expected)

  def test_to_sparse(self):
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(
          [self._filename],
          batch_size=batch_size,
          fields=['col2', 'col0'])
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()
      batch = DataFrame.to_sparse(batch)

    c = self._df['col0']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = DataFrame.Value(
            np.array(expected_values),
            tuple([np.array(expected_splits, dtype=np.int32)]))
        actual = result['col0']
        self.assertAllClose(actual.values, expected.values)
        self.assertAllClose(
            len(set(list(zip(*actual.indices))[0])) + 1,
            len(expected.nested_row_splits[0]))

  def test_map_to_sparse(self):
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(
          [self._filename],
          batch_size=batch_size,
          fields=['col2', 'col0'])
      ds = ds.map(DataFrame.to_sparse)
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    c = self._df['col0']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = DataFrame.Value(
            np.array(expected_values),
            tuple([np.array(expected_splits, dtype=np.int32)]))
        actual = result['col0']
        self.assertAllClose(actual.values, expected.values)
        self.assertAllClose(
            len(set(list(zip(*actual.indices))[0])) + 1,
            len(expected.nested_row_splits[0]))

  def test_apply_to_sparse(self):
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(
          [self._filename],
          batch_size=batch_size,
          fields=['col2', 'col0'])
      ds = ds.apply(to_sparse())
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    c = self._df['col0']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = DataFrame.Value(
            np.array(expected_values),
            tuple([np.array(expected_splits, dtype=np.int32)]))
        actual = result['col0']
        self.assertAllEqual(actual.values, expected.values)
        self.assertAllEqual(
            len(set(list(zip(*actual.indices))[0])) + 1,
            len(expected.nested_row_splits[0]))

  def test_feedable_iterator(self):
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(
          [self._filename],
          batch_size=batch_size,
          fields=['col2', 'col0'])
      ds = ds.apply(to_sparse())
      ds = ds.prefetch(4)
      it = make_one_shot_iterator(ds)
      handle_tensor = it.string_handle()
      feedable_handle = array_ops.placeholder(dtypes.string, shape=[])
      feedable_it = iterator_ops.Iterator.from_string_handle(
          feedable_handle,
          it.output_types,
          it.output_shapes,
          it.output_classes)
      batch = feedable_it.get_next()

    c = self._df['col0']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      handle_val = sess.run(handle_tensor)
      for i in range(3):
        result = sess.run(batch, feed_dict={feedable_handle: handle_val})
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = DataFrame.Value(
            np.array(expected_values),
            tuple([np.array(expected_splits, dtype=np.int32)]))
        actual = result['col0']
        self.assertAllEqual(actual.values, expected.values)
        self.assertAllEqual(
            len(set(list(zip(*actual.indices))[0])) + 1,
            len(expected.nested_row_splits[0]))

  def test_read_and_map(self):
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(
          [self._filename],
          batch_size=batch_size,
          fields=['col2', 'col0'])
      def _parse(values):
        return values['col0'], values['col2']
      ds = ds.map(_parse)
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    c = self._df['col0']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        expected_items = c[start_row:end_row].to_numpy().tolist()
        expected_values = []
        expected_splits = [0]
        for item in expected_items:
          expected_values.extend(item)
          expected_splits.append(expected_splits[-1] + len(item))
        expected = DataFrame.Value(
            np.array(expected_values),
            tuple([np.array(expected_splits, dtype=np.int32)]))
        actual = result[0]
        self.assertAllClose(actual, expected)


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  test.main()

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

r'''Parquet batch dataset test.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import os
import tempfile

from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test

from hybridbackend.tensorflow.data import DataFrame
from hybridbackend.tensorflow.data import make_one_shot_iterator
from hybridbackend.tensorflow.data import ParquetDataset
from hybridbackend.tensorflow.data import read_parquet


# pylint: disable=missing-docstring
class ParquetDatasetTest(test.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    self._workspace = tempfile.mkdtemp()
    self._filename = os.path.join(self._workspace, 'test.parquet')
    self._df = pd.DataFrame(
        np.random.randint(0, 100, size=(200, 4), dtype=np.int64),
        columns=list('ABCD'))
    self._df.to_parquet(self._filename)

  def tearDown(self):  # pylint: disable=invalid-name
    os.remove(self._filename)

  def test_read(self):
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(
          self._filename,
          batch_size=batch_size,
          fields=[DataFrame.Field('A', dtypes.int64),
                  DataFrame.Field('C', dtypes.int64)])
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

  def test_schema_auto_detection_read(self):
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset([self._filename], batch_size=batch_size)
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    c = self._df['C']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        self.assertAllEqual(result['C'], c[start_row:end_row].to_numpy())

  def test_dtype_auto_detection_read(self):
    batch_size = 32
    with ops.Graph().as_default() as graph:
      ds = ParquetDataset(
          [self._filename],
          batch_size=batch_size,
          fields=['B', 'C'])
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    c = self._df['C']
    with self.test_session(use_gpu=False, graph=graph) as sess:
      for i in range(3):
        result = sess.run(batch)
        start_row = i * batch_size
        end_row = (i+1) * batch_size
        self.assertAllEqual(result['C'], c[start_row:end_row].to_numpy())

  def test_read_from_generator(self):
    num_epochs = 2
    batch_size = 100
    with ops.Graph().as_default() as graph:
      def gen_filenames():
        for i in range(num_epochs + 1):
          if i == num_epochs:
            return  # raise StopIteration
          yield self._filename
      filenames = Dataset.from_generator(
          gen_filenames, dtypes.string, tensor_shape.TensorShape([]))
      fields = [
          DataFrame.Field('A', dtypes.int64),
          DataFrame.Field('C', dtypes.int64)]
      ds = filenames.apply(read_parquet(batch_size, fields=fields))
      ds = ds.prefetch(4)
      batch = make_one_shot_iterator(ds).get_next()

    with self.test_session(use_gpu=False, graph=graph) as sess:
      for _ in range(len(self._df) * num_epochs // batch_size):
        sess.run(batch)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(batch)


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  test.main()

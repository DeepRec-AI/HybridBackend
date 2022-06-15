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
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf

import hybridbackend.tensorflow as hb
import hybridbackend.test as hbtest


# pylint: disable=missing-docstring
class ParquetDatasetSequenceRebatchTest(unittest.TestCase):
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    self._workspace = tempfile.mkdtemp()
    self._filename = os.path.join(self._workspace, 'seqtest.parquet')
    self._nrows = 1000
    self._ncols = 10
    self._data = {
      'clicks': [
        [random.randint(0, 100) for col in range(self._ncols)]
        for row in range(self._nrows)]}
    pq.write_table(pa.Table.from_pydict(self._data), self._filename)

  def tearDown(self):  # pylint: disable=invalid-name
    os.remove(self._filename)
    del os.environ['CUDA_VISIBLE_DEVICES']

  def test_ragged(self):
    batch_size = 8
    with tf.Graph().as_default() as graph:
      ds = hb.data.ParquetDataset(self._filename, batch_size=batch_size)
      ds = ds.apply(hb.data.rebatch(batch_size))
      batch = hb.data.make_one_shot_iterator(ds).get_next()

    clicks = self._data['clicks']
    with tf.Session(graph=graph) as sess:
      for i in xrange(3):
        actual = sess.run(batch['clicks'])
        start_row = i * batch_size
        end_row = (i + 1) * batch_size
        expected = clicks[start_row:end_row]
        expected_values = [v for sublist in expected for v in sublist]
        np.testing.assert_equal(actual.values, expected_values)


if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

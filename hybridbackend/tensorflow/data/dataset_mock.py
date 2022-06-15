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

r'''Mock data generator.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm as tq


# pylint: disable=missing-docstring
def generate(params):
  if params.filename:
    filename = params.filename
  else:
    workspace = tempfile.mkdtemp()
    filename = os.path.join(workspace, f'benchmark.{params.format}')

  tf.logging.info('Mock data starts generating...')
  columns = [f'col{c}' for c in xrange(params.cols)]
  df = pd.DataFrame(
    np.random.randint(
      params.min, params.max,
      size=(params.samples, params.cols),
      dtype=np.int64),
    columns=columns)
  if params.format == 'parquet':
    tf.logging.info('Mock data starts writing to parquet...')
    df.to_parquet(filename)
  elif params.format == 'csv':
    tf.logging.info('Mock data starts writing to csv...')
    df.to_csv(filename, header=False, index=False)
  elif params.format == 'tfrecord':
    tf.logging.info('Mock data starts writing to tfrecord...')
    writer = tf.python_io.TFRecordWriter(filename)
    for row in tq(range(params.samples)):
      feats = tf.train.Features(
        feature={
          c: tf.train.Feature(
            int64_list=tf.train.Int64List(value=[df[c][row]]))
          for c in columns})
      example = tf.train.Example(features=feats)
      writer.write(example.SerializeToString())
    writer.close()
  else:
    raise ValueError(f'Format {params.format} not supported')
  tf.logging.info(f'Mock data written to {filename} .')


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--format', type=str, default='parquet')
  parser.add_argument('--samples', type=int, default=20000000)
  parser.add_argument('--cols', type=int, default=200)
  parser.add_argument('--max', type=int, default=100)
  parser.add_argument('--min', type=int, default=0)
  parser.add_argument('filename', default=None)
  generate(parser.parse_args())

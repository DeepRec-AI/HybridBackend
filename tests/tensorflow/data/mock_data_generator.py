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
import numpy as np
import pandas as pd
import os
import tempfile
from tqdm import tqdm as tq

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import tf_logging as logging


# pylint: disable=missing-docstring
def benchmark(params):
  if params.filename:
    filename = params.filename
  else:
    workspace = tempfile.mkdtemp()
    filename = os.path.join(workspace, f'benchmark.{params.format}')

  logging.info('Mock data starts generating...')
  columns = [f'col{c}' for c in range(params.cols)]
  df = pd.DataFrame(
      np.random.randint(
          params.min, params.max,
          size=(params.samples, params.cols),
          dtype=np.int64),
      columns=columns)
  if params.format == 'parquet':
    logging.info('Mock data starts writing to parquet...')
    df.to_parquet(filename)
  elif params.format == 'csv':
    logging.info('Mock data starts writing to csv...')
    df.to_csv(filename, header=False, index=False)
  elif params.format == 'tfrecord':
    logging.info('Mock data starts writing to tfrecord...')
    writer = tf_record.TFRecordWriter(filename)
    for row in tq(range(params.samples)):
      feats = feature_pb2.Features(
          feature={
              c: feature_pb2.Feature(
                  int64_list=feature_pb2.Int64List(value=[df[c][row]]))
              for c in columns})
      example = example_pb2.Example(features=feats)
      writer.write(example.SerializeToString())
    writer.close()
  else:
    raise ValueError(f'Format {params.format} not supported')
  logging.info(f'Mock data written to {filename} .')


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  logging.set_verbosity(logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--format', type=str, default='parquet')
  parser.add_argument('--samples', type=int, default=20000000)
  parser.add_argument('--cols', type=int, default=200)
  parser.add_argument('--max', type=int, default=100)
  parser.add_argument('--min', type=int, default=0)
  parser.add_argument('filename', default=None)
  benchmark(parser.parse_args())

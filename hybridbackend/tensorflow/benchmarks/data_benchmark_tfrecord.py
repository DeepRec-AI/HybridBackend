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

r'''Data reading for TFRecord files benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tempfile
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm as tq


# pylint: disable=missing-docstring
def benchmark(params):
  if not params.filenames:
    tf.logging.info('Started generating mock file ...')
    workspace = tempfile.mkdtemp()
    params.filenames = [os.path.join(workspace, 'benchmark.tfrecord')]
    df = pd.DataFrame(
      np.random.randint(
        0, 100,
        size=(params.batch_size * 100, len(params.fields)),
        dtype=np.int64),
      columns=params.fields)
    writer = tf.python_io.TFRecordWriter(params.filenames[0])
    for row in tq(range(params.samples)):
      feats = tf.train.Features(
        feature={
          f: tf.train.Feature(
            int64_list=tf.train.Int64List(value=[df[f][row]]))
          for f in params.fields})
      example = tf.train.Example(features=feats)
      writer.write(example.SerializeToString())
    writer.close()
    tf.logging.info(f'Mock file {params.filenames[0]} generated.')
  with tf.Graph().as_default():
    step = tf.train.get_or_create_global_step()
    ds = tf.data.TFRecordDataset(params.filenames)
    ds = ds.batch(params.batch_size, drop_remainder=True)
    ds = ds.map(
      lambda line: tf.parse_example(
        line, {f: tf.FixedLenFeature([1], tf.int64) for f in params.fields}))
    batch = tf.data.make_one_shot_iterator(ds).get_next()
    train_op = tf.group(batch + [step.assign_add(1)])
    with tf.train.MonitoredTrainingSession('') as sess:
      count = 0
      prev_ts = time.time()
      try:
        while not sess.should_stop():
          sess.run(train_op)
          count += 1
      except tf.errors.OutOfRangeError:
        pass
      duration = time.time() - prev_ts
      if count <= 0:
        print('Reading TFRecord files stopped unexpectedly')
        return
      print(
        'Reading TFRecord files elapsed in '
        f'{params.batch_size * count / duration:.2f} samples/sec ('
        f'{1000. * duration / count:.2f} msec/step)')


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=64000)
  parser.add_argument(
    '--fields', nargs='+', default=[f'f{c}' for c in xrange(200)])
  parser.add_argument('filenames', nargs='*')
  benchmark(parser.parse_args())

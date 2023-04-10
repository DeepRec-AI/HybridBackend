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

r'''Data reading for Parquet files benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from tqdm import tqdm as tq

import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
def benchmark(params):
  if params.deduplicated_block_size % 2 != 0:
    raise ValueError(
      'deduplicated_block_size must be divided by 2; '
      f'but is {params.deduplicated_block_size}')
  if params.batch_size % params.deduplicated_block_size != 0:
    raise ValueError(
      'params.batch_size must be divided by params.deduplicated_block_size; '
      f'but is {params.batch_size} and {params.deduplicated_block_size}')
  deduplicated_block_num = params.batch_size // params.deduplicated_block_size
  if not params.filenames:
    tf.logging.info('Started generating mock file ...')
    workspace = tempfile.mkdtemp()
    params.filenames = [
      os.path.join(workspace, 'benchmark.parquet'),
      os.path.join(workspace, 'benchmark_deduplicated.parquet')]
    baseline_data = pa.array(
      [[1], [2, 3]] * (params.batch_size * 100 // 2),
      pa.list_(pa.int64()))
    deduplicated_data = pa.array(
      [[[1], [2, 3]] for _ in range(deduplicated_block_num * 100)],
      pa.list_(pa.list_(pa.int64())))
    data_idx = pa.array(
      [[0, 1] * (params.deduplicated_block_size // 2)
       for _ in range(deduplicated_block_num * 100)],
      pa.list_(pa.int64()))

    baseline_table = pa.Table.from_arrays(
      [baseline_data] * len(params.fields),
      names=params.fields)
    pq.write_table(baseline_table, params.filenames[0], compression='ZSTD')
    tf.logging.info(f'Mock file {params.filenames[0]} generated.')

    deduplicated_table = pa.Table.from_arrays(
      [deduplicated_data] * len(params.fields) + [data_idx],
      names=(params.fields + ['data_idx']))
    pq.write_table(deduplicated_table, params.filenames[1], compression='ZSTD')
    tf.logging.info(f'Mock file {params.filenames[1]} generated.')

  with tf.Graph().as_default():
    step = tf.train.get_or_create_global_step()
    if params.baseline:
      ds = hb.data.Dataset.from_parquet([params.filenames[0]])
      ds = ds.batch(params.batch_size, drop_remainder=True)
    else:
      ds = hb.data.Dataset.from_parquet(
        [params.filenames[1]],
        key_idx_field_names=['data_idx'],
        value_field_names=[params.fields])
      ds = ds.batch(deduplicated_block_num, drop_remainder=True)
    batch = tf.data.make_one_shot_iterator(ds).get_next()
    train_op = tf.group(list(batch.values()) + [step.assign_add(1)])
    with tf.train.MonitoredTrainingSession('') as sess:
      count = 0
      prev_ts = time.time()
      try:
        with tq() as pbar:
          should_stop = False
          while not sess.should_stop() and not should_stop:
            prev_sess_run = time.time()
            sess.run(train_op)
            sess_run_duration = time.time() - prev_sess_run
            pbar.set_description(
              f'{params.batch_size / sess_run_duration:6.2f} samples/sec')
            pbar.update(1)
            count += 1
            if params.num_steps is not None:
              should_stop = count >= params.num_steps
      except tf.errors.OutOfRangeError:
        pass
      duration = time.time() - prev_ts
      if count <= 0:
        print('Reading Parquet files stopped unexpectedly')
        return
      print(
        'Reading Parquet files elapsed in '
        f'{params.batch_size * count / duration:.2f} samples/sec ('
        f'{1000. * duration / count:.2f} msec/step)')


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline', default=False, action='store_true')
  parser.add_argument('--deduplicate', default=False, action='store_true')
  parser.add_argument('--batch-size', type=int, default=64000)
  parser.add_argument('--deduplicated-block-size', type=int, default=1000)
  parser.add_argument('--num-steps', type=int, default=None)
  parser.add_argument(
    '--fields', nargs='+', default=[f'f{c}' for c in xrange(20)])
  parser.add_argument('filenames', nargs='*')
  benchmark(parser.parse_args())

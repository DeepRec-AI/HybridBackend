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

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm as tq

import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
def benchmark(params):
  if not params.filenames:
    tf.logging.info('Started generating mock file ...')
    workspace = tempfile.mkdtemp()
    params.filenames = [os.path.join(workspace, 'benchmark.parquet')]
    if params.use_string_data:
      df = pd.DataFrame(
        np.array([
          [
            *[
              np.array(list(map(str, np.random.randint(
                0, 9,
                size=(np.random.randint(10, 30),),
                dtype=np.int64))))
              for _ in xrange(len(params.fields))]]
          for _ in xrange(params.batch_size * 100)], dtype=object),
        columns=params.fields)
    elif params.use_fixed_len_string_data:
      df = pd.DataFrame(
        np.array([
          ['abcdefghijklmnoprstu' for _ in xrange(len(params.fields))]
          for _ in xrange(params.batch_size * 100)], dtype=np.str),
        columns=params.fields)
    else:
      df = pd.DataFrame(
        np.random.randint(
          0, 100,
          size=(params.batch_size * 100, len(params.fields)),
          dtype=np.int64),
        columns=params.fields)
    df.to_parquet(params.filenames[0])
    tf.logging.info(f'Mock file {params.filenames[0]} generated.')
  with tf.Graph().as_default():
    step = tf.train.get_or_create_global_step()
    if params.baseline:
      ds = hb.data.Dataset.from_parquet(params.filenames)
      ds = ds.map(lambda data: data)  # Prevent fusion
      if params.shuffle:
        ds = ds.shuffle(params.batch_size * 10)
      ds = ds.batch(params.batch_size, drop_remainder=True)
    else:
      ds = hb.data.Dataset.from_parquet(params.filenames)
      if params.shuffle:
        ds = ds.shuffle_batch(
          params.batch_size, drop_remainder=True,
          buffer_size=params.batch_size * 10)
      else:
        ds = ds.batch(params.batch_size, drop_remainder=True)
    batch = tf.data.make_one_shot_iterator(ds).get_next()
    train_op = tf.group(list(batch.values()) + [step.assign_add(1)])
    chief_only_hooks = []
    if params.profile_every_n_iter is not None:
      chief_only_hooks.append(
        tf.train.ProfilerHook(
          save_steps=params.profile_every_n_iter,
          output_dir=params.output_dir))
    with tf.train.MonitoredTrainingSession(
        '', chief_only_hooks=chief_only_hooks) as sess:
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
  parser.add_argument('--shuffle', default=False, action='store_true')
  parser.add_argument('--use-string-data', default=False, action='store_true')
  parser.add_argument(
    '--use-fixed-len-string-data', default=False, action='store_true')
  parser.add_argument('--batch-size', type=int, default=64000)
  parser.add_argument('--num-steps', type=int, default=None)
  parser.add_argument('--output-dir', default='./outputs')
  parser.add_argument('--profile-every-n-iter', type=int, default=None)
  parser.add_argument(
    '--fields', nargs='+', default=[f'f{c}' for c in xrange(200)])
  parser.add_argument('filenames', nargs='*')
  benchmark(parser.parse_args())

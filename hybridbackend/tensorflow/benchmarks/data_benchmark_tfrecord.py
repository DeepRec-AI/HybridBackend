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
    writer = tf.python_io.TFRecordWriter(params.filenames[0])
    for row in tq(range(params.batch_size * 100)):
      if params.use_string_data or params.use_fixed_len_string_data:
        feats = tf.train.Features(
          feature={
            f: tf.train.Feature(
              bytes_list=tf.train.BytesList(
                value=[bytes(val, 'utf-8') for val in df[f][row]]))
            for f in params.fields})
      else:
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
    if params.shuffle:
      ds = ds.shuffle(params.batch_size * 10)
    ds = ds.batch(params.batch_size, drop_remainder=True)
    if params.use_string_data or params.use_fixed_len_string_data:
      ds = ds.map(
        lambda line: tf.parse_example(
          line, {f: tf.VarLenFeature(tf.string) for f in params.fields}))
    else:
      ds = ds.map(
        lambda line: tf.parse_example(
          line, {f: tf.FixedLenFeature([1], tf.int64) for f in params.fields}))
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

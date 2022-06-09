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

r'''Data reading benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
def describe_csv():
  return [[1 << 32] for f in xrange(200)]


def describe_tfrecord():
  return {
    f'col{c}': tf.FixedLenFeature([1], tf.int64)
    for c in xrange(200)}


def benchmark(params):
  with tf.Graph().as_default():
    step = tf.train.get_or_create_global_step()
    if params.filename.endswith('.parquet'):
      ds = hb.data.ParquetDataset(
        [params.filename] * params.epochs,
        batch_size=params.batch_size)
      if params.rebatch_size is not None:
        ds = ds.apply(hb.data.rebatch(params.rebatch_size))
      batch = hb.data.make_one_shot_iterator(ds).get_next()
      count_op = tf.shape(list(batch.values())[0])[0]
      train_op = tf.group(list(batch.values()) + [step.assign_add(1)])
    elif params.filename.endswith('.csv'):
      ds = tf.data.TextLineDataset([params.filename] * params.epochs)
      ds = ds.batch(params.batch_size)
      ds = ds.map(lambda line: tf.io.decode_csv(line, describe_csv()))
      batch = hb.data.make_one_shot_iterator(ds).get_next()
      count_op = tf.shape(batch[0])[0]
      train_op = tf.group(batch + [step.assign_add(1)])
    elif params.filename.endswith('.tfrecord'):
      ds = tf.data.TFRecordDataset([params.filename] * params.epochs)
      ds = ds.batch(params.batch_size)
      ds = ds.map(lambda line: tf.parse_example(line, describe_tfrecord()))
      batch = hb.data.make_one_shot_iterator(ds).get_next()
      count_op = tf.shape(batch[0])[0]
      train_op = tf.group(batch + [step.assign_add(1)])
    else:
      raise ValueError(f'File {params.filename} not supported.')
    with hb.train.monitored_session(
        hooks=[
          tf.train.StopAtStepHook(params.num_steps),
          hb.train.StepStatHook(count=count_op)]) as sess:
      while not sess.should_stop():
        sess.run(train_op)


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  os.environ['MALLOC_CONF'] = 'background_thread:true,metadata_thp:auto'
  os.environ['ARROW_NUM_THREADS'] = '16'
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=200000)
  parser.add_argument('--rebatch-size', type=int, default=None)
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument('--num-steps', type=int, default=100)
  parser.add_argument('filename')
  benchmark(parser.parse_args())

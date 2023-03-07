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

r'''Embedding training using unobtrusive API on single GPU benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import tensorflow as tf

import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
def benchmark(params):
  with tf.Graph().as_default():
    fields = hb.data.Dataset.schema_from_parquet(params.filenames[0])
    fields = [
      f for f in fields
      if f.name not in ('label', 'ts') and f.dtype in (tf.int32, tf.int64)]
    ds = hb.data.Dataset.from_parquet(params.filenames, fields=fields)
    ds = ds.batch(params.batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    iterator = tf.data.make_one_shot_iterator(ds)
    iterator = hb.data.Iterator(iterator)
    iterator_hook = hb.data.Iterator.Hook()
    inputs = iterator.get_next()
    outputs = []
    with tf.name_scope('features'):
      for field in inputs:
        with tf.device('/cpu:0'):
          embedding_weights = tf.get_variable(
            f'{field}_weight',
            shape=(128, 32),
            initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
        ids = inputs[field]
        if isinstance(ids, tf.Tensor):
          ids = ids % params.dimension_size
          ids, idx = tf.unique(tf.reshape(ids, shape=[-1]))
          embeddings = tf.nn.embedding_lookup(embedding_weights, ids)
          embeddings = tf.gather(embeddings, idx)
        else:
          ids = tf.SparseTensor(
            ids.indices,
            ids.values % params.dimension_size,
            ids.dense_shape)
          embeddings = tf.nn.embedding_lookup_sparse(
            embedding_weights, ids, None)
        outputs.append(embeddings)
    loss = tf.math.add_n([tf.reduce_sum(t) for t in outputs])
    opt = hb.train.AdagradOptimizer(learning_rate=0.01)
    step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=step)

    with tf.train.MonitoredTrainingSession('', hooks=[iterator_hook]) as sess:
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
        print('Training embedding layers stopped unexpectedly')
        return
      print(
        'Training embedding layers elapsed in '
        f'{params.batch_size * count / duration:.2f} samples/sec ('
        f'{1000. * duration / count:.2f} msec/step)')


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  os.environ['HB_OP_RELOCATION_ENABLED'] = '1'
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=16384)
  parser.add_argument('--dimension-size', type=int, default=32)
  parser.add_argument('filenames', nargs='+')
  benchmark(parser.parse_args())

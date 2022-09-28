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

r'''Recmendation model training benchmark using DenseFeatures.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
@hb.function()
def build_bench_op(params):
  fields = params.fields.split(',')
  with tf.device('/cpu:0'):
    ds = hb.data.ParquetDataset(
      params.filenames,
      fields=fields,
      batch_size=params.batch_size,
      num_parallel_reads=len(params.filenames))
    ds = ds.apply(hb.data.parse())

  ds = ds.prefetch(params.num_steps)
  features = hb.data.make_one_shot_iterator(ds).get_next()
  columns = {
    f: tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_identity(
        key=f,
        num_buckets=params.bucket_size,
        default_value=0),
      dimension=params.dimension,
      initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
    for f in fields}
  embedding_lookup = hb.keras.layers.DenseFeatures(columns.values())
  col_to_embeddings = {}
  embedding_lookup(features, col_to_embeddings)
  embeddings = [
    col_to_embeddings[columns[f]]
    for f in fields]
  loss = tf.math.add_n([tf.reduce_sum(emb) for emb in embeddings])
  opt = hb.train.AdamOptimizer(learning_rate=params.lr)
  step = tf.train.get_or_create_global_step()
  return opt.minimize(loss, global_step=step)


def benchmark(params):
  bench_op = build_bench_op(params)
  hooks = [tf.train.StopAtStepHook(params.num_steps)]
  hooks.append(
    hb.train.StepStatHook(count=params.batch_size, unit='sample'))
  if params.profile:
    hooks.append(
      tf.train.ProfilerHook(
        save_steps=params.num_steps - 1,
        output_dir='.'))
  with hb.train.monitored_session(hooks=hooks) as sess:
    while not sess.should_stop():
      sess.run(bench_op)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', type=float, default=1.)
  parser.add_argument('--dimension', type=int, default=32)
  parser.add_argument('--bucket-size', type=int, default=100000)
  parser.add_argument('--batch-size', type=int, default=100000)
  parser.add_argument('--num-steps', type=int, default=100)
  parser.add_argument('--profile', default=False, action='store_true')
  parser.add_argument('--fields', type=str)
  parser.add_argument('filenames', nargs='+')
  benchmark(parser.parse_args())

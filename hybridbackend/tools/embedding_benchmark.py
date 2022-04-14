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

r'''Embedding benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import tensorflow as tf

import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
def build_bench_op(params, categorical_fields):
  with tf.device('/cpu:0'):
    ds = hb.data.ParquetDataset(
      params.filenames,
      fields=categorical_fields.keys(),
      batch_size=params.batch_size,
      num_parallel_reads=len(params.filenames))
    ds = ds.apply(hb.data.to_sparse())
  with tf.device(hb.train.device_setter()):
    ds = ds.prefetch(params.num_steps)
    features = hb.data.make_one_shot_iterator(ds).get_next()
    columns = {
      fid: hb.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
          key=fid,
          num_buckets=categorical_fields[fid],
          default_value=0),
        dimension=params.dimension,
        initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
      for fid in categorical_fields}
    embedding_lookup = hb.feature_column.DenseFeatures(columns.values())
    col_to_embeddings = {}
    embedding_lookup(features, col_to_embeddings)
    embeddings = [
      col_to_embeddings[columns[f]]
      for f in categorical_fields]
    loss = tf.math.add_n([tf.reduce_sum(emb) for emb in embeddings])
    opt = hb.train.AdamOptimizer(learning_rate=params.lr)
    step = tf.train.get_or_create_global_step()
    return opt.minimize(loss, global_step=step)


def benchmark(params):
  with tf.io.gfile.GFile(params.config_file, 'rb') as f:
    config = json.loads(f.read().decode('utf-8'))
  categorical_fields = config[params.config_file_field]
  if params.max_bucket_size is not None:
    mbs = params.max_bucket_size
    categorical_fields = {
      f: mbs if categorical_fields[f] > mbs else categorical_fields[f]
      for f in categorical_fields}

  with tf.device(hb.train.device_setter()):
    bench_op = build_bench_op(params, categorical_fields)
  hooks = [tf.train.StopAtStepHook(params.num_steps)]
  hooks.append(
    hb.train.StepStatHook(count=params.batch_size, unit='sample'))
  if params.profile:
    hooks.append(
      tf.train.ProfilerHook(
        save_steps=params.num_steps - 1,
        output_dir='.'))
  server = hb.train.Server()
  with hb.train.MonitoredTrainingSession(
      server.target,
      is_chief=True,
      hooks=hooks) as sess:
    while not sess.should_stop():
      sess.run(bench_op)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', type=float, default=1.)
  parser.add_argument('--dimension', type=int, default=32)
  parser.add_argument('--max-bucket-size', type=int, default=1000000)
  parser.add_argument('--batch-size', type=int, default=100000)
  parser.add_argument('--num-steps', type=int, default=10)
  parser.add_argument('--profile', default=False, action='store_true')
  parser.add_argument('--config-file-field', type=str, default='categorical')
  parser.add_argument('--config-file', type=str)
  parser.add_argument('filenames', nargs='+')
  benchmark(parser.parse_args())

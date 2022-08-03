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

r'''Recmendation model training benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import re

import tensorflow as tf

import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
def build_bench_op(params):
  with tf.device('/cpu:0'):
    ds = hb.data.ParquetDataset(
      params.filenames,
      batch_size=params.batch_size,
      num_parallel_reads=len(params.filenames),
      drop_remainder=True)
    ds = ds.apply(hb.data.to_sparse())

    def map_fn(batch):
      labels = {}
      numerics = {}
      ids = {}
      for f in batch:
        if re.match(params.label_field_pattern, f):
          labels[f] = tf.to_float(batch[f])
        elif re.match(params.numeric_field_pattern, f):
          numerics[f] = tf.to_float(batch[f])
        elif re.match(params.categorical_field_pattern, f):
          sp_ids = batch[f]
          if isinstance(sp_ids, tf.Tensor):
            sp_ids = tf.sparse.from_dense(
              tf.reshape(sp_ids, (params.batch_size, 1)))
          ids[f] = tf.SparseTensor(
            sp_ids.indices,
            sp_ids.values % params.bucket_size,
            sp_ids.dense_shape)
      return labels, numerics, ids

    ds = ds.map(map_fn)
    ds = ds.prefetch(1)
    iterator = tf.data.make_one_shot_iterator(ds)
  if not params.noprefetch:
    iterator = hb.data.Iterator(iterator)
  labels, numerics, ids = iterator.get_next()
  embedding_weights = {}
  embeddings = {}
  for f, v in ids.items():
    embedding_weight_name = f'{f}_weight'
    with tf.device('/cpu:0'):
      embedding_weights[embedding_weight_name] = tf.get_variable(
        f'{f}_weight',
        shape=(params.bucket_size, params.dimension),
        initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
    embeddings[f] = tf.nn.safe_embedding_lookup_sparse(
      embedding_weights[embedding_weight_name], v,
      default_id=params.default_id)
  labels = tf.concat(
    [tf.reshape(v, [-1, 1]) for v in labels.values()],
    axis=1)
  numerics = tf.concat(
    [tf.reshape(v, [-1, 1]) for v in numerics.values()],
    axis=1)
  embeddings = tf.concat(
    [tf.reshape(v, [-1, params.dimension]) for v in embeddings.values()],
    axis=1)
  mlp_dims = [int(d) for d in params.mlp_dims.split(',') if d]
  mlp_input = tf.concat([numerics, embeddings], axis=1)

  for i, d in enumerate(mlp_dims):
    mlp_input = tf.layers.dense(
      mlp_input, d,
      activation=tf.nn.relu,
      kernel_initializer=tf.random_normal_initializer(
        mean=0.0,
        stddev=math.sqrt(2.0 / d)),
      bias_initializer=tf.random_normal_initializer(
        mean=0.0,
        stddev=math.sqrt(1.0 / d)),
      name=f'mlp_{i}')
  logits = tf.layers.dense(
    mlp_input, 1,
    activation=tf.nn.sigmoid,
    kernel_initializer=tf.random_normal_initializer(
      mean=0.0,
      stddev=math.sqrt(2.0)),
    bias_initializer=tf.random_normal_initializer(
      mean=0.0,
      stddev=math.sqrt(1.0)),
    name='logits')

  loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
  mlp_weights = {v.name: v for v in tf.trainable_variables()}
  for _, v in embedding_weights.items():
    del mlp_weights[v.name]

  step = tf.train.get_or_create_global_step()

  embedding_opt = tf.train.AdagradOptimizer(learning_rate=params.lr)
  embedding_train_op = embedding_opt.minimize(
    loss, global_step=step, var_list=list(embedding_weights.values()))

  mlp_opt = tf.train.AdamOptimizer(learning_rate=params.lr)
  mlp_train_op = mlp_opt.minimize(
    loss, global_step=step, var_list=list(mlp_weights.values()))

  return tf.group([embedding_train_op, mlp_train_op])


def benchmark(params):
  with tf.device('/gpu:0'):
    bench_op = build_bench_op(params)
  hooks = [tf.train.StopAtStepHook(params.num_steps)]
  hooks.append(
    hb.train.StepStatHook(count=params.batch_size, unit='sample'))
  if not params.noprefetch:
    hooks.append(hb.data.Iterator.Hook())
  if params.profile:
    hooks.append(
      tf.train.ProfilerHook(
        save_steps=params.num_steps - 1,
        output_dir='.'))
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  config.gpu_options.force_gpu_compatible = True
  with tf.train.MonitoredTrainingSession(
      '', hooks=hooks, config=config) as sess:
    while not sess.should_stop():
      sess.run(bench_op)


if __name__ == '__main__':
  os.environ['TF_CPP_VMODULE'] = (
    'optimize_embedding_ops=2,'
    'optimize_unique=2,'
    'optimize_sparse_segment_reduction=2,'
    'optimize_sparse_fill_empty_rows=2,'
    'replacing=2,'
    'horizontal_fusion=2,'
    'pruning=2,'
    'relocation=2')
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', type=float, default=1.)
  parser.add_argument('--dimension', type=int, default=32)
  parser.add_argument('--bucket-size', type=int, default=100000)
  parser.add_argument('--batch-size', type=int, default=100000)
  parser.add_argument('--num-steps', type=int, default=100)
  parser.add_argument('--profile', default=False, action='store_true')
  parser.add_argument('--label-field-pattern', default='^label$')
  parser.add_argument('--categorical-field-pattern', default='^id.*')
  parser.add_argument('--numeric-field-pattern', default='^f.*')
  parser.add_argument('--categorical-field-missing-value', type=int, default=0)
  parser.add_argument('--default-id', type=int, default=None)
  parser.add_argument('--mlp-dims', default='')
  parser.add_argument('--noprefetch', default=False, action='store_true')
  parser.add_argument('filenames', nargs='+')
  benchmark(parser.parse_args())

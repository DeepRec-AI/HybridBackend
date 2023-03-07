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

r'''Pipeline benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math

import tensorflow as tf

import hybridbackend.tensorflow as hb


# pylint: disable=missing-docstring
def build_bench_op(params):
  fields = params.fields.split(',')
  with tf.device('/cpu:0'):
    ds = hb.data.Dataset.from_parquet(
      params.filenames,
      fields=fields)
    ds = ds.batch(params.batch_size)

  @hb.pipeline.compute_pipeline()
  def build_layers(features):
    columns = [
      tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
          key=f,
          num_buckets=params.bucket_size,
          default_value=0),
        dimension=params.dimension,
        initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
      for f in fields]
    embeddings = [
      tf.feature_column.input_layer(features, [c]) for c in columns]
    num_s_features = len(embeddings)
    mlp_amp_sizes = [2048] * params.amp_dense_layer_num
    mlp_sizes = [1024, 1024, 512, 256, 1]

    with tf.name_scope('dot_interact'):
      dot_interact_input = tf.concat(embeddings, axis=-1)
      dot_interact_input = tf.reshape(
        dot_interact_input, [-1, num_s_features, params.dimension])
      x2 = tf.matmul(dot_interact_input, dot_interact_input, transpose_b=True)
      x2_dim = x2.shape[-1]
      x2_ones = tf.ones_like(x2)
      x2_mask = tf.linalg.band_part(x2_ones, 0, -1)
      dot_interact_output = tf.boolean_mask(x2, x2_ones - x2_mask)
      dot_interact_output = tf.reshape(
        dot_interact_output, [-1, x2_dim * (x2_dim - 1) // 2])

    with tf.name_scope('mlp'):
      mlp_input = tf.concat([dot_interact_output], axis=1)
      prev_d = (num_s_features * num_s_features) / 2 + params.dimension
      for i, d in enumerate(mlp_amp_sizes):
        mlp_input = tf.layers.dense(
          mlp_input,
          d,
          activation=tf.nn.relu,
          kernel_initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=math.sqrt(2.0 / (prev_d + d))),
          bias_initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=math.sqrt(1.0 / d)),
          trainable=False,
          name=f'mlp_amp_{i}')
        prev_d = d
      for i, d in enumerate(mlp_sizes[:-1]):
        mlp_input = tf.layers.dense(
          mlp_input, d,
          activation=tf.nn.relu,
          kernel_initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=math.sqrt(2.0 / (prev_d + d))),
          bias_initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=math.sqrt(1.0 / d)),
          name=f'mlp_{i}')
        prev_d = d
      mlp_output = tf.layers.dense(
        mlp_input, mlp_sizes[-1],
        activation=tf.nn.sigmoid,
        kernel_initializer=tf.random_normal_initializer(
          mean=0.0,
          stddev=math.sqrt(2.0 / (prev_d + mlp_sizes[-1]))),
        bias_initializer=tf.random_normal_initializer(
          mean=0.0,
          stddev=math.sqrt(1.0 / mlp_sizes[-1])),
        name=f'mlp_{len(mlp_sizes) - 1}')
    return tf.reduce_sum(mlp_output)

  features = tf.data.make_one_shot_iterator(ds).get_next()
  loss = build_layers(features)
  opt = tf.train.GradientDescentOptimizer(learning_rate=params.lr)
  step = tf.train.get_or_create_global_step()
  return opt.minimize(loss, global_step=step)


def benchmark(params):
  with hb.scope(data_batch_count=2):
    bench_op = build_bench_op(params)
    hooks = [tf.train.StopAtStepHook(params.num_steps)]
    hooks.append(
      hb.train.StepStatHook(
        every_n_iter=10,
        count=params.batch_size * params.data_batch_count,
        unit='sample'))
    if params.profile:
      hooks.append(tf.train.ProfilerHook(
        save_steps=params.num_steps - 1,
        output_dir='.'))
    with tf.train.MonitoredTrainingSession('', hooks=hooks) as sess:
      while not sess.should_stop():
        sess.run(bench_op)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--amp-dense-layer-num', type=int, default=5)
  parser.add_argument('--lr', type=float, default=1.)
  parser.add_argument('--dimension', type=int, default=32)
  parser.add_argument('--bucket-size', type=int, default=100000)
  parser.add_argument('--batch-size', type=int, default=100000)
  parser.add_argument('--data-batch-count', type=int, default=2)
  parser.add_argument('--num-steps', type=int, default=100)
  parser.add_argument('--profile', default=False, action='store_true')
  parser.add_argument('--fields', type=str, default='')
  parser.add_argument('filenames', nargs='+')
  benchmark(parser.parse_args())

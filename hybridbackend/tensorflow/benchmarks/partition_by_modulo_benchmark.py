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

r'''Benchmark for partition, lookup and stitch.
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
def shard_lookup(weight_list, ids_list, params, fusion):
  results = []
  if fusion == 'disabled':
    for i, weight in enumerate(weight_list):
      with tf.name_scope(f'input_{i}'):
        ids = ids_list[i]
        ids_shards, idx_shards = hb.distribute.partition(
          ids, params.num_partitions)
        embs_shards = [
          tf.gather(weight, ids_shards[i] % params.bucket_size)
          for i in xrange(params.num_partitions)]
        results.append(tf.dynamic_stitch(idx_shards, embs_shards))
  else:
    for i, weight in enumerate(weight_list):
      with tf.name_scope(f'input_{i}'):
        ids = ids_list[i]
        ids, _, idx = hb.distribute.partition_by_modulo(
          ids, params.num_partitions)
        embs = tf.gather(weight, ids % params.bucket_size)
        results.append(tf.gather(embs, idx))
  return results


def build_bench_op(params):
  value_limit = 3 * params.num_columns * params.column_ids
  with tf.device(params.device):
    weight_initializer = tf.random_uniform(
      [params.bucket_size, params.dim_size], 0, 100.0,
      dtype=tf.float32)
    weights = [
      tf.get_variable(
        f'weight{c}',
        use_resource=False,
        dtype=tf.float32,
        initializer=weight_initializer)
      for c in xrange(params.num_columns)]
  input_initializer = tf.random_uniform(
    [params.column_ids], 0, value_limit, dtype=tf.int32)
  inputs = [
    tf.get_variable(
      f'input{c}',
      use_resource=False,
      dtype=tf.int32,
      initializer=input_initializer)
    for c in xrange(params.num_columns)]
  bench_ops = shard_lookup(
    weights, inputs, params,
    fusion=params.fusion)
  bench_ops = [tf.math.reduce_sum(o) for o in bench_ops]
  step = tf.train.get_or_create_global_step()
  bench_ops.append(tf.cast(step.assign_add(1), tf.float32))
  bench_op = tf.math.add_n(bench_ops)
  return bench_op


@hb.function()
def benchmark(params):
  bench_op = build_bench_op(params)
  hooks = [tf.train.StopAtStepHook(params.num_steps)]
  hooks.append(
    hb.train.StepStatHook(
      count=params.column_ids * params.num_columns / 1000000.,
      unit='Mlookups'))
  with tf.train.MonitoredTrainingSession('', hooks=hooks) as sess:
    while not sess.should_stop():
      sess.run(bench_op)


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', type=str, default='/gpu:0')
  parser.add_argument('--column-ids', type=int, default=100000)
  parser.add_argument('--dim-size', type=int, default=32)
  parser.add_argument('--bucket-size', type=int, default=10000)
  parser.add_argument('--num-columns', type=int, default=100)
  parser.add_argument('--num-partitions', type=int, default=8)
  parser.add_argument('--num-steps', type=int, default=10)
  parser.add_argument('--fusion', type=str, default='vertical_and_horizontal')
  benchmark(parser.parse_args())

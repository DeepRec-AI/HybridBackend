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

r'''Transfer benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client.timeline import Timeline

import hybridbackend.tensorflow as hb  # pylint: disable=unused-import # noqa: F401


# pylint: disable=missing-docstring
def _uniform_sizes(total_size, num_partitions):
  uniform_distro = [1. / num_partitions for _ in range(num_partitions)]
  return np.random.multinomial(total_size, uniform_distro, size=1)[0].tolist()


def host_to_device(message_floats, message_partitions, random_sizes=True):
  if random_sizes:
    partition_counts = _uniform_sizes(message_floats, message_partitions)
  else:
    partition_counts = [
      int(message_floats / message_partitions)
      for _ in range(message_partitions)]
  h2d_outputs = []
  for p in range(message_partitions):
    message_count = partition_counts[p]
    with tf.device('/cpu:0'):
      h2d_input = tf.get_variable(
        f'input{p}',
        initializer=tf.random.normal(
          [message_count],
          mean=100,
          stddev=80),
        use_resource=False)
    with tf.device('/gpu:0'):
      h2d_outputs.append(tf.identity(h2d_input))
  with tf.device('/gpu:0'):
    bench_op = tf.math.reduce_sum(tf.concat(h2d_outputs, 0))
  return bench_op, lambda sess, op: sess.run(op)


def unpin_host_to_device(message_floats, message_partitions, random_sizes=True):
  if random_sizes:
    partition_counts = _uniform_sizes(message_floats, message_partitions)
  else:
    partition_counts = [
      int(message_floats / message_partitions)
      for _ in range(message_partitions)]
  h2d_inputs = []
  h2d_input_values = []
  h2d_outputs = []
  for p in range(message_partitions):
    message_count = partition_counts[p]
    h2d_input_values.append(np.empty([message_count], dtype=np.float32))
    with tf.device('/cpu:0'):
      h2d_input = tf.placeholder(
        tf.float32,
        [message_count])
      h2d_inputs.append(h2d_input)
    with tf.device('/gpu:0'):
      h2d_outputs.append(tf.identity(h2d_input))
  with tf.device('/gpu:0'):
    bench_op = tf.math.reduce_sum(tf.concat(h2d_outputs, 0))

  def _call(sess, op):
    return sess.run(
      op,
      feed_dict={
        h2d_inputs[p]: h2d_input_values[p]
        for p in range(message_partitions)})
  return bench_op, _call


def benchmark(args):
  transfer_ops = {
    'H2D_': lambda mf, mp: host_to_device(mf, mp, random_sizes=False),
    'H2D': host_to_device,
    'UH2D_': lambda mf, mp: unpin_host_to_device(mf, mp, random_sizes=False),
    'UH2D': unpin_host_to_device}
  for cop in args.transfer_ops:
    if cop not in transfer_ops:
      raise ValueError(
        f'Specified transfer op type `{cop}` not in {transfer_ops.keys()}')
  for ns in args.message_sizes:
    for part in args.message_partitions:
      if ns * 262144 % part != 0:
        raise ValueError(
          f'{ns * 262144} floats cannot be divided into {part} partitions')

  with tf.Graph().as_default():
    bench_ops = {
      cop: {
        ns: {part: {} for part in args.message_partitions}
        for ns in args.message_sizes}
      for cop in args.transfer_ops}
    for cop in args.transfer_ops:
      with tf.name_scope(cop), tf.variable_scope(cop):
        for ns in args.message_sizes:
          with tf.name_scope(f'{ns}mb'), tf.variable_scope(f'{ns}mb'):
            for p in args.message_partitions:
              with tf.name_scope(f'{p}parts'), tf.variable_scope(f'{p}parts'):
                bench_ops[cop][ns][p] = transfer_ops[cop](ns * 262144, p)
    profile_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    with tf.train.MonitoredTrainingSession('') as sess:
      print('Op\tTotal Size\t#Splits\tDuration\tThroughput')
      for cop in args.transfer_ops:
        for ns in args.message_sizes:
          for part in args.message_partitions:
            bench_op, bench_call = bench_ops[cop][ns][part]
            prev_ts = time.time()
            for _ in range(args.num_steps):
              bench_call(sess, bench_op)
            duration = time.time() - prev_ts
            if duration > 0:
              avg_duration = 1. * duration / args.num_steps
              print(
                f'{cop}\t'
                f'{ns:6.2f}MB\t{part}\t'
                f'{avg_duration * 1000:6.2f}ms\t'
                f'{ns / avg_duration:6.2f}MB/s')
            else:
              print(
                f'{cop}\t'
                f'{ns:6.2f}MB\t{part}\t'
                f'   NaN\t'
                f'   NaN')
            if args.profile:
              profile_metadata = tf.RunMetadata()
              sess.run(
                bench_op,
                options=profile_options,
                run_metadata=profile_metadata)
              ts = Timeline(profile_metadata.step_stats)
              chrome_trace = ts.generate_chrome_trace_format(
                show_dataflow=False, show_memory=False)
              profile_fname = (
                f'./transfer-benchmark-timeline-{cop}-{ns}mb-{part}p.json')
              with open(profile_fname, 'w', encoding='utf-8') as f:
                f.write(chrome_trace)


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', default=False, action='store_true')
  parser.add_argument(
    '--transfer-ops',
    nargs='+',
    help='Transfer ops in (H2D,)',
    default=['H2D'])
  parser.add_argument(
    '--message-sizes',
    type=int,
    nargs='+',
    help='Total size (MB) of messages to transfer',
    default=[32, 64, 128, 256, 512, 1024])
  parser.add_argument(
    '--message-partitions',
    type=int,
    nargs='+',
    help='Number of partitions of each message',
    default=[1, 8, 64, 512])
  parser.add_argument('--num-steps', type=int, default=10)
  benchmark(parser.parse_args())

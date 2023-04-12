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

r'''Collective benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import tensorflow as tf

import hybridbackend.tensorflow as hb  # pylint: disable=unused-import # noqa: F401


# pylint: disable=missing-docstring
def allreduce(message_floats, message_partitions, message_device, topology):
  del topology
  step = tf.train.get_or_create_global_step()
  results = [step.assign_add(1)]
  for p in range(message_partitions):
    with tf.device(f'/{message_device}:0'):
      coll_input = tf.get_variable(
        f'input{p}',
        initializer=tf.random.normal(
          [int(message_floats / message_partitions)],
          mean=100,
          stddev=80))
    coll_output = hb.distribute.allreduce(coll_input)
    with tf.device(f'/{message_device}:0'):
      results.append(tf.identity(coll_output))
  return tf.group(results)


def alltoall(message_floats, message_partitions, message_device):
  step = tf.train.get_or_create_global_step()
  results = [step.assign_add(1)]
  for p in range(message_partitions):
    with tf.device(f'/{message_device}:0'):
      coll_input = tf.get_variable(
        f'input{p}',
        initializer=tf.random.normal(
          [int(message_floats / message_partitions)],
          mean=100,
          stddev=80))
    coll_output = hb.distribute.alltoall(coll_input)
    with tf.device(f'/{message_device}:0'):
      results.append(tf.identity(coll_output))
  return tf.group(results)


def _uniform_sizes(total_size, active_size):
  uniform_distro = [
    1. / active_size for _ in range(active_size)]
  return np.random.multinomial(total_size, uniform_distro, size=1)[0].tolist()


def alltoallv(
    message_floats, message_partitions, message_device, topology,
    random_sizes=True):
  active_size = hb.distribute.active_size(topology)
  step = tf.train.get_or_create_global_step()
  results = [step.assign_add(1)]
  for p in range(message_partitions):
    with tf.device(f'/{message_device}:0'):
      message_count = int(message_floats / message_partitions)
      coll_input = tf.get_variable(
        f'input{p}',
        initializer=tf.random.normal(
          [message_count],
          mean=100,
          stddev=80))
      if random_sizes:
        coll_input_sizes = tf.constant(
          _uniform_sizes(message_count, active_size),
          dtype=tf.int32)
      else:
        message_divided = message_count // active_size
        coll_input_sizes = tf.constant(
          [message_divided for _ in range(active_size)],
          dtype=tf.int32)
    coll_output, _ = hb.distribute.alltoall(
      coll_input, sizes=coll_input_sizes, topology=topology)
    with tf.device(f'/{message_device}:0'):
      results.append(tf.identity(coll_output))
  return tf.group(results)


def benchmark(args):
  collective_ops = {
    'allreduce': allreduce,
    'alltoall': alltoall,
    'alltoallv_': lambda mf, mp, md, topo: alltoallv(
      mf, mp, md, topo, random_sizes=False),
    'alltoallv': alltoallv}
  for cop in args.collective_ops:
    if cop not in collective_ops:
      raise ValueError(
        f'Specified collective op type `{cop}` '
        f'not in {collective_ops.keys()}')
  for nf in args.message_floats:
    if nf % hb.context.world_size != 0:
      raise ValueError(
        f'#floats {nf} cannot be divided onto {hb.context.world_size} devices')
    for part in args.message_partitions:
      if nf % part != 0:
        raise ValueError(
          f'#floats {nf} cannot be divided into {part} partitions')

  with tf.Graph().as_default(), hb.scope():
    bench_ops = {
      cop: {
        nf: {
          part: {
            dev: {} for dev in args.message_devices}
          for part in args.message_partitions}
        for nf in args.message_floats}
      for cop in args.collective_ops}
    for cop in args.collective_ops:
      with tf.name_scope(cop), tf.variable_scope(cop):
        for nf in args.message_floats:
          with tf.name_scope(f'{nf}floats'), tf.variable_scope(f'{nf}floats'):
            for p in args.message_partitions:
              with tf.name_scope(f'{p}parts'), tf.variable_scope(f'{p}parts'):
                for dev in args.message_devices:
                  with tf.name_scope(f'{dev}'), tf.variable_scope(f'{dev}'):
                    for topo in args.collective_topology:
                      with tf.name_scope(f'{topo}topology'), tf.variable_scope(
                          f'{topo}topology'):
                        bench_ops[cop][nf][p][dev][topo] = (
                          collective_ops[cop](
                            nf, p, dev, topo))
    with tf.train.MonitoredTrainingSession('') as sess:
      print('Rank\tCollective\tTopology\tDevice\tSize\t#Splits\tThroughput')
      # pylint: disable=too-many-nested-blocks
      for cop in args.collective_ops:
        for dev in args.message_devices:
          for nf in args.message_floats:
            for part in args.message_partitions:
              for topo in args.collective_topology:
                for _ in range(args.warmup_steps):
                  sess.run(bench_ops[cop][nf][part][dev][topo])
                prev_ts = time.time()
                for _ in range(args.num_steps):
                  sess.run(bench_ops[cop][nf][part][dev][topo])
                duration = time.time() - prev_ts
                message_mbs = nf * 4. / 1024. / 1024.
                print(
                  f'{hb.context.rank}/{hb.context.world_size}\t'
                  f'{cop}\tTopology-{topo}\t{dev}\t'
                  f'{message_mbs:.2f}MB\t{part}\t'
                  f'{args.num_steps * message_mbs * 8.0 / duration:.2f}Gb/s')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--collective-ops',
    nargs='+',
    help='Collective ops in (allreduce, alltoall, alltoallv)',
    default=['allreduce', 'alltoall', 'alltoallv'])
  parser.add_argument(
    '--collective-topology',
    type=int,
    nargs='+',
    help='All/Intra/Inter nodes participate collective_ops',
    default=[0, 1, 2])
  parser.add_argument(
    '--message-floats',
    type=int,
    nargs='+',
    help='Count of floats in each message',
    default=[65536, 262144, 1048576, 4194304, 16777216])
  parser.add_argument(
    '--message-partitions',
    type=int,
    nargs='+',
    help='Number of partitions of each message',
    default=[1, 8, 64])
  parser.add_argument(
    '--message-devices',
    nargs='+',
    help='Number of devices of each message',
    default=['gpu', 'cpu'])
  parser.add_argument('--warmup-steps', type=int, default=100)
  parser.add_argument('--num-steps', type=int, default=500)
  benchmark(parser.parse_args())

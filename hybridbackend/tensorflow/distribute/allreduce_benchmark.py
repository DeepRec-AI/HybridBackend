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

r'''AllReduce benchmark.
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
  comm_input = tf.get_variable(
    'input',
    initializer=tf.random_normal(
      [params.kilobytes * 256],
      mean=100,
      stddev=80))
  comm = hb.distribute.Communicator.build(
    'benchmark', hb.context.devices)
  comm_output = comm.allreduce(comm_input)
  step = tf.train.get_or_create_global_step()
  return tf.group([comm_output, step.assign_add(1)])


def benchmark(params):
  with hb.scope():
    bench_op = build_bench_op(params)
    with tf.train.MonitoredTrainingSession(
        '',
        hooks=[
          tf.train.StopAtStepHook(params.num_steps),
          hb.train.StepStatHook(
            count=params.kilobytes / 1024.,
            unit='MB')]) as sess:
      while not sess.should_stop():
        sess.run(bench_op)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--kilobytes', type=int, default=1024 * 64)  # 64MB
  parser.add_argument('--num-steps', type=int, default=1000)
  benchmark(parser.parse_args())

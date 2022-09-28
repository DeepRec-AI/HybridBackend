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

r'''Functions for optimization
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def lr_with_linear_warmup_and_polynomial_decay(
    global_step,
    initial_value=24.,
    scaling_factor=1.,
    warmup_steps=None,
    decay_steps=None,
    decay_start_step=None,
    decay_exp=2,
    epsilon=1.e-7):
  r'''Calculates learning rate with linear warmup and polynomial decay.

  Args:
    global_step: Variable representing the current step.
    initial_value: Initial value of learning rates.
    warmup_steps: Steps of warmup.
    decay_steps: Steps of decay.
    decay_start_step: Start step of decay.
    decay_exp: Exponent part of decay.
    scaling_factor: Factor for scaling.

  Returns:
    New learning rate tensor.
  '''
  initial_lr = tf.constant(initial_value * scaling_factor, tf.float32)

  if warmup_steps is None:
    return initial_lr

  global_step = tf.cast(global_step, tf.float32)
  warmup_steps = tf.constant(warmup_steps, tf.float32)
  warmup_rate = initial_lr / warmup_steps
  warmup_lr = initial_lr - (warmup_steps - global_step) * warmup_rate

  if decay_steps is None or decay_start_step is None:
    return warmup_lr

  decay_start_step = tf.constant(decay_start_step, tf.float32)
  steps_since_decay_start = global_step - decay_start_step
  decay_steps = tf.constant(decay_steps, tf.float32)
  decayed_steps = tf.minimum(steps_since_decay_start, decay_steps)
  to_decay_rate = (decay_steps - decayed_steps) / decay_steps
  decay_lr = initial_lr * to_decay_rate**decay_exp
  decay_lr = tf.maximum(decay_lr, tf.constant(epsilon))

  warmup_lambda = tf.cast(global_step < warmup_steps, tf.float32)
  decay_lambda = tf.cast(global_step > decay_start_step, tf.float32)
  initial_lambda = tf.cast(
    tf.math.abs(warmup_lambda + decay_lambda) < epsilon, tf.float32)

  lr = warmup_lambda * warmup_lr
  lr += decay_lambda * decay_lr
  lr += initial_lambda * initial_lr
  return lr


def sgd_decay_optimize(
    loss,
    lr_initial_value,
    lr_warmup_steps,
    lr_decay_start_step,
    lr_decay_steps):
  r'''Optimize using SGD and learning rate decay.
  '''
  step = tf.train.get_or_create_global_step()
  lr = lr_with_linear_warmup_and_polynomial_decay(
    step,
    initial_value=lr_initial_value,
    warmup_steps=lr_warmup_steps,
    decay_start_step=lr_decay_start_step,
    decay_steps=lr_decay_steps)
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  return opt.minimize(loss, global_step=step)

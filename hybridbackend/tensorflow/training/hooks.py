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

r'''Session run hook for training.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import summary_io
from tensorflow.python.training import training_util


class Policy(session_run_hook.SessionRunHook):
  r'''Policy for training.
  '''
  class Trigger(session_run_hook.SessionRunHook):
    r'''Session run hook to execute training policies after specific step.
    '''
    def __init__(self, policies, scaffold=None, output_dir=None):
      r'''Initializes a `Policy`.

      Args:
        policies: `list` of TrainingPolicy.
        scaffold: Scaffold from Monitored session.
        output_dir: `str`, base directory for the summary files.
      '''
      self._policies = policies
      self._scaffold = scaffold or monitored_session.Scaffold()

      metrics = {}
      for p in policies:
        metrics.update(p.metrics)

      self._tag_order = sorted(metrics.keys())
      self._metrics = metrics
      self._output_dir = output_dir
      self._summary_writer = None
      self._timers = []
      for p in policies:
        self._timers.append(
          basic_session_run_hooks.NeverTriggerTimer() if p.only_at_end
          else basic_session_run_hooks.SecondOrStepTimer(
            every_secs=p.every_n_secs,
            every_steps=p.every_n_steps))
      self._steps_per_run = 1
      self._iter = 0

    def begin(self):
      r'''Called once before using the session.
      '''
      for t in self._timers:
        t.reset()
      self._current_metrics = {
        tag: basic_session_run_hooks._as_graph_element(tensor)  # pylint: disable=protected-access
        for (tag, tensor) in self._metrics.items()}
      self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
      if self._global_step_tensor is None:
        raise RuntimeError('Global step should be created to use TrainingHook.')
      if ops.GraphKeys.GLOBAL_STEP not in self._current_metrics:
        self._current_metrics[ops.GraphKeys.GLOBAL_STEP] = (
          self._global_step_tensor)
      self._summary_writer = None
      if self._output_dir:
        self._summary_writer = summary_io.SummaryWriterCache.get(
          self._output_dir)

    def after_create_session(self, session, coord):
      r'''Called when new TensorFlow session is created.
      '''
      _ = coord
      global_step = session.run(self._global_step_tensor)
      if self._summary_writer:
        graph = ops.get_default_graph()
        self._summary_writer.add_graph(graph)
        saver_def = None
        if self._scaffold.saver:
          saver_def = self._scaffold.saver.saver_def
        meta_graph_def = meta_graph.create_meta_graph_def(
          graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def)
        self._summary_writer.add_meta_graph(meta_graph_def)

      for t in self._timers:
        t.update_last_triggered_step(global_step)
      self._iter = 0

    def before_run(self, run_context):
      r'''Called before each call to run().
      '''
      _ = run_context
      return session_run_hook.SessionRunArgs(self._current_metrics)

    def after_run(self, run_context, run_values):
      r'''Called after each call to run().
      '''
      stale_global_step = run_values.results[ops.GraphKeys.GLOBAL_STEP]
      should_trigger = any(
        t.should_trigger_for_step(stale_global_step + self._steps_per_run)
        for t in self._timers)
      if should_trigger:
        global_step = run_context.session.run(self._global_step_tensor)
        for idx, policy in enumerate(self._policies):
          if self._timers[idx].should_trigger_for_step(global_step):
            ctx_stop_requested = run_context.stop_requested
            run_context._stop_requested = False  # pylint: disable=protected-access
            values = {k: run_values.results[k] for k in policy.metrics}
            values.pop(ops.GraphKeys.GLOBAL_STEP, None)
            if 'iter' not in values:
              values['iter'] = self._iter
            policy(
              global_step, values, run_context.session,
              context=run_context,
              scaffold=self._scaffold,
              writer=self._summary_writer)
            self._timers[idx].update_last_triggered_step(global_step)
            run_context._stop_requested = ctx_stop_requested  # pylint: disable=protected-access
      self._iter += 1

    def end(self, session):
      r'''Called at the end of session.
      '''
      at_end = any(p.at_end for p in self._policies)
      if at_end:
        last_step = session.run(self._global_step_tensor)
        should_trigger = any(
          last_step != t.last_triggered_step()
          for t in self._timers)
        if should_trigger:
          all_values = session.run(self._current_metrics)
          for idx, policy in enumerate(self._policies):
            if last_step != self._timers[idx].last_triggered_step():
              values = {k: all_values[k] for k in policy.metrics}
              values.pop(ops.GraphKeys.GLOBAL_STEP, None)
              self._policies[idx](
                last_step, values, session,
                context=None,
                scaffold=self._scaffold,
                writer=self._summary_writer)

  def __init__(
      self,
      metrics=None,
      every_n_steps=None,
      every_n_secs=None,
      at_end=False):
    r'''Initializes a `Policy`.

    Args:
      metrics: `dict` that maps string-valued tags to tensors/tensor names, or
        `iterable` of tensors/tensor names.
      every_n_steps: `int`, calls every N global steps.
      every_n_secs: `int` or `float`, calls every N seconds. Exactly
        one of `every_n_steps` and `every_n_secs` should be provided.
      at_end: `bool` specifying whether to calls at the end of the run.
    '''
    self._only_at_end = (
      at_end and (every_n_steps is None) and (every_n_secs is None))
    if (not self._only_at_end
        and (every_n_steps is None) == (every_n_secs is None)):
      raise ValueError(
        'either at_end and/or exactly one of every_n_steps and every_n_secs '
        'must be provided.')
    if not isinstance(metrics, dict):
      raise ValueError('metrics should be a `dict`.')

    self._metrics = metrics or {}
    self._every_n_steps = every_n_steps
    self._every_n_secs = every_n_secs
    self._at_end = at_end

  def __call__(
      self, global_step, metrics, session,
      context=None,
      scaffold=None,
      writer=None):
    r'''Function called after a training step.

    Args:
      global_step: Global step value.
      metrics: Metric values.
      session: Session for running extra steps.
      context: Run context for this policy.
      scaffold: Scaffold from Monitored session.
      writer: Writer to write summaries.
    '''
    raise NotImplementedError

  @property
  def metrics(self):
    return self._metrics

  @property
  def every_n_steps(self):
    return self._every_n_steps

  @property
  def every_n_secs(self):
    return self._every_n_secs

  @property
  def at_end(self):
    return self._at_end

  @property
  def only_at_end(self):
    return self._only_at_end


class StepStatHook(session_run_hook.SessionRunHook):
  r'''Hook that counts performance statistics for steps.
  '''
  def __init__(self, every_n_iter=None, count=None, unit='sample'):
    self._every_n_iter = every_n_iter
    self._count = count
    self._unit = unit
    self._count_runnable = isinstance(self._count, (ops.Operation, ops.Tensor))

  @property
  def should_print_logs(self):
    return (
      self._every_n_iter is not None
      and self._iter_count > 0
      and self._iter_count % self._every_n_iter == 0)

  def begin(self):
    r'''Called once before using the session.
    '''
    self._iter_count = 0
    self._durations = []
    self._counts = []
    self._prev_ts = None

  def before_run(self, run_context):
    r'''Called before each call to run().
    '''
    _ = run_context
    self._prev_ts = time.time()
    if self._count_runnable:
      return session_run_hook.SessionRunArgs(self._count)
    return None

  def after_run(self, run_context, run_values):
    r'''Called after each call to run().
    '''
    self._durations.append(time.time() - self._prev_ts)
    _ = run_context
    if self._count is not None:
      if self._count_runnable:
        self._counts.append(run_values.results)
      else:
        self._counts.append(self._count)
      if self.should_print_logs:
        durs = np.array(self._durations)
        cnts = np.array(self._counts)
        p50 = np.percentile(durs, 50)
        p10 = np.percentile(durs, 10)
        p90 = np.percentile(durs, 90)
        fps = 1. * np.sum(cnts) / np.sum(durs)
        logging.info(
          f'secs/step: {p50:.5f} ({100.*p10/p90:.2f}%), '
          f'{self._unit}s/sec: {fps:.2f}')
        self._durations = []
        self._counts = []
    else:
      if self.should_print_logs:
        durs = np.array(self._durations)
        p50 = np.percentile(durs, 50)
        p10 = np.percentile(durs, 10)
        p90 = np.percentile(durs, 90)
        logging.info(f'secs/step: {p50:.5f} ({100.*p10/p90:.2f}%)')
        self._durations = []
    self._iter_count += 1

  def end(self, session):
    r'''Called at the end of session.
    '''
    _ = session
    if self._count is not None:
      durs = np.array(self._durations)
      cnts = np.array(self._counts)
      if self._durations:
        p50 = np.percentile(durs, 50)
        p10 = np.percentile(durs, 10)
        p90 = np.percentile(durs, 90)
        fps = 1. * np.sum(cnts) / np.sum(durs)
        logging.info(
          f'secs/step: {p50:.5f} ({100.*p10/p90:.2f}%), '
          f'{self._unit}s/sec: {fps:.2f}')
      self._durations = []
      self._counts = []
    else:
      durs = np.array(self._durations)
      if self._durations:
        p50 = np.percentile(durs, 50)
        p10 = np.percentile(durs, 10)
        p90 = np.percentile(durs, 90)
        logging.info(f'secs/step: {p50:.5f} ({100.*p10/p90:.2f}%)')
      self._durations = []

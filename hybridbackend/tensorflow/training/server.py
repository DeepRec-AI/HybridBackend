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

r'''Servers using hybrid parallelism.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session as _monitored_session
from tensorflow.python.training import server_lib

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.context import context_scope
from hybridbackend.tensorflow.framework.device import device_function
from hybridbackend.tensorflow.training.evaluation import EvaluationHook
from hybridbackend.tensorflow.training.function import configure


def wraps_monitored_session(
    cls, keep_checkpoint_max=5, keep_checkpoint_every_n_hours=10000.0):
  r'''Decorator to create wrapped monitored session.
  '''
  class _WrappedMonitoredSession(cls):
    r'''Session-like object that handles initialization, recovery and hooks.
    '''
    def __init__(self, session_creator=None, hooks=None, **kwargs):
      r'''Creates a new WrappedMonitoredSession.
      '''
      with context_scope(
          keep_checkpoint_max=keep_checkpoint_max,
          keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours):
        for h in hooks:
          if isinstance(h, basic_session_run_hooks.CheckpointSaverHook):
            h._listeners += Context.get().saving_listeners
        with ops.device('/cpu:0'):
          super(cls, self).__init__(  # pylint: disable=bad-super-call
            session_creator, hooks, should_recover=True, **kwargs)

    def _make_callable_from_options(self, callable_options):
      return self._sess._sess._sess._sess._make_callable_from_options(  # pylint: disable=protected-access
        callable_options)

  return _WrappedMonitoredSession


def wraps_server(cls):
  r'''Decorator to create hybridbackend server class.
  '''
  class HybridBackendServer(cls):
    r'''An in-process TensorFlow server, for use in distributed training.
    '''
    _default = None

    @classmethod
    def get(class_):
      if class_._default is None:
        class_._default = class_()
      return class_._default

    def __init__(self, server_or_cluster_def=None, **kwargs):
      r'''Creates a new server with the given definition.
      '''
      if server_or_cluster_def is None:
        server_or_cluster_def = Context.get().cluster_spec
      if server_or_cluster_def is None:
        self._is_local = True
        return
      self._is_local = False
      kwargs['job_name'] = Context.get().task_type
      kwargs['task_index'] = Context.get().task_id
      kwargs['config'] = configure(prototype=kwargs.pop('config', None))
      super().__init__(server_or_cluster_def, **kwargs)

    @property
    def target(self):
      r'''Returns the target for asession to connect to this server.
      '''
      if self._is_local:
        return ''
      return super().target

    def monitored_session(self, **kwargs):
      r'''Creates a `MonitoredSession` for training.
      '''
      ctx = Context.get()
      checkpoint_dir = kwargs.get('checkpoint_dir', None)
      summary_dir = kwargs.get('summary_dir', None)
      summary_dir = summary_dir or checkpoint_dir
      eval_dir = None
      if summary_dir is not None:
        eval_dir = os.path.join(
          summary_dir,
          f'eval_{ctx.rank}' if ctx.world_size > 1 else 'eval')
      hooks = kwargs.pop('hooks', [])
      hooks.extend(ctx.training_hooks)
      eval_every_n_iter = kwargs.pop('eval_every_n_iter', None)
      eval_steps = kwargs.pop('eval_steps', 100)
      eval_fn = kwargs.pop('eval_fn', None)
      if eval_every_n_iter is not None:
        if eval_fn is None:
          raise ValueError('eval_fn must be specified')

        def _eval_fn():
          r'''Actual evaluation function.
          '''
          with context_scope(model_dir=checkpoint_dir):
            eval_metric_ops = eval_fn()
          if not isinstance(eval_metric_ops, dict):
            raise ValueError('eval_fn should return a dict of metric ops')
          update_ops = []
          metrics = {}
          for metric_name, metric_val_and_update in eval_metric_ops.items():
            if not isinstance(metric_name, str):
              raise ValueError(f'Metric name {metric_name} should be a str')
            if (not isinstance(metric_val_and_update, (tuple, list))
                or len(metric_val_and_update) != 2):
              raise ValueError(
                f'{metric_val_and_update} should be (metric, update_op)')
            update_ops.append(metric_val_and_update[0])
            metrics[metric_name] = metric_val_and_update[1]
          update_op = control_flow_ops.group(update_ops)
          return update_op, metrics, None
        hooks.append(
          EvaluationHook(
            _eval_fn,
            steps=eval_steps,
            every_n_iter=eval_every_n_iter,
            summary_dir=eval_dir))
      kwargs['hooks'] = hooks
      kwargs['config'] = configure(prototype=kwargs.pop('config', None))
      with ops.device(device_function), context_scope(
          model_dir=checkpoint_dir):
        prev_monitored_session = _monitored_session.MonitoredSession
        _monitored_session.MonitoredSession = wraps_monitored_session(
          prev_monitored_session,
          keep_checkpoint_max=kwargs.pop('keep_checkpoint_max', 5),
          keep_checkpoint_every_n_hours=kwargs.pop(
            'keep_checkpoint_every_n_hours', 10000.0))
        sess = _monitored_session.MonitoredTrainingSession(
          master=self.target, is_chief=True, **kwargs)
        _monitored_session.MonitoredSession = prev_monitored_session
        return sess

  return HybridBackendServer


Server = wraps_server(server_lib.Server)


def monitored_session(**kwargs):
  r'''Creates a `MonitoredSession` for training with default server.
  '''
  return Server.get().monitored_session(**kwargs)

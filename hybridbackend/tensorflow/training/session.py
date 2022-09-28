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
from six import string_types as string

from tensorflow._api.v1 import train as train_v1
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session as _monitored_session
from tensorflow.python.training import training

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.context import context_scope
from hybridbackend.tensorflow.framework.device import device_function
from hybridbackend.tensorflow.training.config import configure
from hybridbackend.tensorflow.training.eval import EvaluationHook
from hybridbackend.tensorflow.training.function import Patching
from hybridbackend.tensorflow.training.policy import Policy
from hybridbackend.tensorflow.training.server import Server


class HybridBackendMonitoredSessionBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of monitored session wrapper.
  '''


def wraps_monitored_session(
    cls, keep_checkpoint_max=5, keep_checkpoint_every_n_hours=10000.0):
  r'''Decorator to create wrapped monitored session.
  '''
  if issubclass(cls, HybridBackendMonitoredSessionBase):
    return cls

  class HybridBackendMonitoredSession(cls, HybridBackendMonitoredSessionBase):
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

  return HybridBackendMonitoredSession


def wraps_monitored_training_session(fn):
  r'''Decorator to create wrapped monitored training session.
  '''
  if hasattr(fn, 'wrapped_fn'):
    return fn

  def HybridBackendMonitoredTrainingSession(*args, **kwargs):  # pylint: disable=invalid-name
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
    scaffold = kwargs.pop('scaffold', _monitored_session.Scaffold())
    kwargs['scaffold'] = scaffold
    hooks = kwargs.pop('hooks', [])
    hooks.extend(ctx.training_hooks)
    policies = [h for h in hooks if isinstance(h, Policy)]
    if policies:
      hooks.append(
        Policy.Trigger(
          policies,
          scaffold=scaffold,
          output_dir=summary_dir))
    chief_only_hooks = kwargs.pop('chief_only_hooks', [])
    chief_only_hooks.extend(ctx.training_chief_hooks)
    chief_only_policies = [
      h for h in chief_only_hooks if isinstance(h, Policy)]
    if chief_only_policies:
      chief_only_hooks.append(
        Policy.Trigger(
          chief_only_policies,
          scaffold=scaffold,
          output_dir=summary_dir))
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
          if not isinstance(metric_name, string):
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
    kwargs['chief_only_hooks'] = chief_only_hooks
    kwargs['config'] = configure(prototype=kwargs.pop('config', None))
    kwargs['is_chief'] = True
    args = list(args)
    if args:
      master = args[0]
      if not master:
        master = Server.get().target
      args[0] = master
    else:
      master = kwargs.pop('master', None)
      if not master:
        master = Server.get().target
      kwargs['master'] = master

    with ops.device(device_function), context_scope(
        model_dir=checkpoint_dir):
      prev_monitored_session = _monitored_session.MonitoredSession
      _monitored_session.MonitoredSession = wraps_monitored_session(
        prev_monitored_session,
        keep_checkpoint_max=kwargs.pop('keep_checkpoint_max', 5),
        keep_checkpoint_every_n_hours=kwargs.pop(
          'keep_checkpoint_every_n_hours', 10000.0))
      sess = fn(*args, **kwargs)
      _monitored_session.MonitoredSession = prev_monitored_session
      return sess

  HybridBackendMonitoredTrainingSession.wrapped_fn = fn
  return HybridBackendMonitoredTrainingSession


class PatchingSession(Patching):
  r'''Patching monitored training session.
  '''
  def __init__(self):
    super().__init__()
    self._prev_sess_fn = None

  def patch(self):
    r'''Patches APIs.
    '''
    self._prev_sess_fn = _monitored_session.MonitoredTrainingSession
    _monitored_session.MonitoredTrainingSession = (
      wraps_monitored_training_session(
        _monitored_session.MonitoredTrainingSession))
    training.MonitoredTrainingSession = (
      _monitored_session.MonitoredTrainingSession)
    train_v1.MonitoredTrainingSession = (
      _monitored_session.MonitoredTrainingSession)

  def unpatch(self):
    r'''Revert API patching.
    '''
    train_v1.MonitoredTrainingSession = self._prev_sess_fn
    training.MonitoredTrainingSession = self._prev_sess_fn
    _monitored_session.MonitoredTrainingSession = self._prev_sess_fn


Patching.register(PatchingSession)

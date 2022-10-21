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

r'''Evaluation related functions.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import six

from google.protobuf import message
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as core_summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import moving_averages
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import slot_creator
from tensorflow.python.training import training_util

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.device import device_function
from hybridbackend.tensorflow.framework.ops import ModeKeys
from hybridbackend.tensorflow.framework.rewriting import GraphRewriting
from hybridbackend.tensorflow.framework.rewriting import scope
from hybridbackend.tensorflow.training.variables import disable_variable_update
from hybridbackend.tensorflow.training.variables import reuse_variables


class EvaluationRewriting(GraphRewriting):
  r'''Rewriting evaluation.
  '''
  def __init__(self, ):
    super().__init__()
    self._prev_create_slot_var = None
    self._prev_assign_moving_average = None
    self._patched = False

  def wraps_create_slot_var(self, fn):
    r'''wraps create slot var to eliminate out scope
    '''
    def wrapped_create_slot_var(primary, *args, **kwargs):
      variable_scope_name = vs.get_variable_scope()._name  # pylint: disable=protected-access
      if (not isinstance(primary, variables.Variable)
          and ModeKeys.EVAL in variable_scope_name):
        vs.get_variable_scope()._name = variable_scope_name.replace(  # pylint: disable=protected-access
          ModeKeys.EVAL, '')
      return fn(primary, *args, **kwargs)
    return wrapped_create_slot_var

  def wraps_assign_moving_average(self, assign_moving_avg_fn):
    r'''disable the update of slot variables within moving average
    '''
    def wrapped_assign_moving_average(
        variable, value, decay, zero_debias=True, name=None):
      with disable_variable_update():
        return assign_moving_avg_fn(
          variable, value, decay, zero_debias=zero_debias, name=name)
    return wrapped_assign_moving_average

  def begin(self):
    r'''Rewrites API.
    '''
    if Context.get().options.mode == ModeKeys.EVAL:
      self._prev_create_slot_var = slot_creator._create_slot_var  # pylint: disable=protected-access
      slot_creator._create_slot_var = self.wraps_create_slot_var(  # pylint: disable=protected-access
        self._prev_create_slot_var)
      self._prev_assign_moving_average = moving_averages.assign_moving_average
      moving_averages.assign_moving_average = self.wraps_assign_moving_average(
        self._prev_assign_moving_average)
      self._patched = True

  def end(self):
    r'''Revert API rewriting.
    '''
    if self._patched:
      slot_creator._create_slot_var = self._prev_create_slot_var  # pylint: disable=protected-access
      moving_averages.assign_moving_average = self._prev_assign_moving_average


GraphRewriting.register(EvaluationRewriting)


EvaluationSpec = collections.namedtuple(
  'EvaluationSpec', ['name', 'hooks', 'update_op', 'eval_dict'])


class EvaluationHook(session_run_hook.SessionRunHook):
  r'''Hook to make evaluation along with training.
  '''
  def __init__(
      self, fn,
      steps=100,
      every_n_iter=1000,
      history=None):
    r'''Evaluates specific function.

    Args:
      fn: Function returns update_op, metric ops and hooks.
      steps: Number of steps for which to evaluate model.
      every_n_iter: `int`, runs the evaluator once every N training iteration.
      history: History of eval metrics. history should support `append` method.
    Raises:
      ValueError: if `every_n_iter` is non-positive or it's not a single machine
        training
    '''
    if every_n_iter is None or every_n_iter <= 0:
      raise ValueError(f'invalid every_n_iter={every_n_iter}.')
    self._fn = fn
    self._steps = steps
    self._every_n_iter = every_n_iter
    self._history = history

  def begin(self):
    r'''Preprocess global step and evaluation's hooks.
    '''
    self._evaluation_specs = ops.get_collection_ref(EvaluationSpec.__name__)
    if len(self._evaluation_specs) > 0:
      raise ValueError('Only one evaluation spec allowed in a graph')

    self._timer = None
    self._iter_count = 0
    self._hooks = []

    self._timer = basic_session_run_hooks.SecondOrStepTimer(
      every_steps=self._every_n_iter)
    self._timer.reset()
    self._summary_dir = Context.get().options.eval_dir or '.'

    with ops.device(device_function), reuse_variables(vs.AUTO_REUSE):
      with scope(mode=ModeKeys.EVAL, comm_pool_name=ModeKeys.EVAL):
        with ops.name_scope(ModeKeys.EVAL):
          eval_spec = self._fn()
          if isinstance(eval_spec, dict):
            eval_dict = {}
            update_ops = []
            for metric_name, metric_val_and_update in eval_spec.items():
              if not isinstance(metric_name, six.string_types):
                raise ValueError(f'Metric name {metric_name} should be a str')
              if (not isinstance(metric_val_and_update, (tuple, list))
                  or len(metric_val_and_update) != 2):
                raise ValueError(
                  f'{metric_val_and_update} should be a tuple '
                  'of (metric, update_op)')
              eval_dict[metric_name] = metric_val_and_update[0]
              update_ops.append(metric_val_and_update[1])
            update_op = control_flow_ops.group(update_ops)
            eval_spec = EvaluationSpec(
              name=EvaluationSpec.__name__,
              hooks=None,
              update_op=update_op,
              eval_dict=eval_dict)
          if not isinstance(eval_spec, EvaluationSpec):
            raise ValueError('eval_fn should return a dict or a EvaluationSpec')
          self._evaluation_specs.append(eval_spec)
          if eval_spec.hooks:
            self._hooks.extend(eval_spec.hooks)
          eval_dict = dict(eval_spec.eval_dict)
          if ops.GraphKeys.GLOBAL_STEP not in eval_dict:
            global_step_tensor = training_util.get_global_step(
              ops.get_default_graph())
            eval_dict[ops.GraphKeys.GLOBAL_STEP] = global_step_tensor
          for h in self._hooks:
            h.begin()
          self._update_op = eval_spec.update_op
          self._metrics = eval_dict

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    r'''Call evaluation's hooks.
    '''
    if ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS):
      raise ValueError(
        'EvaluationHook does not support saveables other than global '
        'variables.')
    for h in self._hooks:
      h.after_create_session(session, coord)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    r'''Runs evaluator after session run.
    '''
    self._iter_count += 1
    if self._timer.should_trigger_for_step(self._iter_count):
      ctx_stop_requested = run_context.stop_requested
      run_context._stop_requested = False  # pylint: disable=protected-access
      self._evaluate(run_context)
      run_context._stop_requested = ctx_stop_requested  # pylint: disable=protected-access

  def _call_before_run_hooks(
      self, run_context, fetch_dict, user_feed_dict=None):
    r'''Call hooks.before_run and handle requests from hooks.
    '''
    hook_feeds = {}
    for hook in self._hooks:
      request = hook.before_run(run_context)
      if request is not None:
        if request.fetches is not None:
          fetch_dict[hook] = request.fetches
        if request.feed_dict:
          hook_feeds.update(request.feed_dict)

    if not hook_feeds:
      return user_feed_dict

    if not user_feed_dict:
      return hook_feeds

    hook_feeds.update(user_feed_dict)
    return hook_feeds

  def _run(self, run_context, fetches):
    r'''Run the evaluation.
    '''
    if isinstance(fetches, dict):
      actual_fetches = fetches
    else:
      actual_fetches = {fetches: fetches}
    eval_metrics = self._call_before_run_hooks(
      run_context, actual_fetches)
    eval_results = run_context.session.run(
      actual_fetches, feed_dict=eval_metrics)
    for hook in self._hooks:
      hook.after_run(
        run_context,
        session_run_hook.SessionRunValues(
          results=eval_results.pop(hook, None),
          options=config_pb2.RunOptions(),
          run_metadata=config_pb2.RunMetadata()))
    return eval_results

  def _write_dict_to_summary(self, dictionary):
    r'''Write evaluation results to eval_dir.
    '''
    current_global_step = dictionary[ops.GraphKeys.GLOBAL_STEP]
    prev_np_printoptions = np.get_printoptions()
    np.set_printoptions(suppress=True)
    stats = ', '.join(
      f'{k} = {v}'
      for k, v in sorted(six.iteritems(dictionary))
      if not (
        isinstance(v, six.binary_type)
        or k == ops.GraphKeys.GLOBAL_STEP))
    np.set_printoptions(**prev_np_printoptions)
    logging.info('Saving metrics for step %d: %s', current_global_step, stats)

    summary_dir = self._summary_dir
    if Context.get().world_size > 1:
      summary_dir = os.path.join(summary_dir, f'{Context.get().rank}')
    summary_writer = core_summary.FileWriterCache.get(summary_dir)
    summary_proto = summary_pb2.Summary()

    for key in dictionary:
      if dictionary[key] is None:
        continue
      if key == 'global_step':
        continue
      if isinstance(dictionary[key], (np.float32, float)):
        summary_proto.value.add(tag=key, simple_value=float(dictionary[key]))
      elif isinstance(dictionary[key], (np.int64, np.int32, int)):
        summary_proto.value.add(tag=key, simple_value=int(dictionary[key]))
      elif isinstance(dictionary[key], six.binary_type):
        try:
          summ = summary_pb2.Summary.FromString(dictionary[key])
          for i, _ in enumerate(summ.value):
            summ.value[i].tag = f'{key}/{i}'
          summary_proto.value.extend(summ.value)
        except message.DecodeError:
          logging.warning(
            'Skipping summary for %s, cannot parse string to Summary.', key)
          continue
      elif isinstance(dictionary[key], np.ndarray):
        value = summary_proto.value.add()
        value.tag = key
        value.node_name = key
        tensor_proto = tensor_util.make_tensor_proto(dictionary[key])
        value.tensor.CopyFrom(tensor_proto)
        # pylint: disable=line-too-long
        logging.info(
          'Summary for np.ndarray is not visible in Tensorboard by default. '
          'Consider using a Tensorboard plugin for visualization (see '
          'https://github.com/tensorflow/tensorboard-plugin-example/blob/master/README.md'
          ' for more information).')
        # pylint: enable=line-too-long
      else:
        logging.warning(
          'Skipping summary for %s, must be a float, np.float32, np.int64, '
          'np.int32 or int or np.ndarray or a serialized string of Summary.',
          key)
    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()

  def _evaluate(self, run_context):
    r'''Evaluate on run context.
    '''
    for _ in range(self._steps):
      if not run_context.stop_requested:
        self._run(run_context, self._update_op)
    metric_values = self._run(run_context, self._metrics)
    if metric_values is not None:
      if self._history is not None:
        self._history.append(metric_values)
      self._write_dict_to_summary(metric_values)
    self._timer.update_last_triggered_step(self._iter_count)

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

r'''Session run hook for parallel evaluation along side of training.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from google.protobuf import message

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as core_summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

from hybridbackend.tensorflow.framework.context import Context


class ParallelEvaluationHook(session_run_hook.SessionRunHook):
  r'''Hook to do a parallel evaluation along side of HB's training
  '''
  def __init__(self,
               update_op,
               eval_dict,
               hooks=None,
               steps=10,
               every_n_iter=100,
               eval_dir=None):
    r'''Initializes a `ParallelEvaluationHook`.

    Args:
      hooks: List of `SessionRunHook` subclass instances. Used for callbacks
        inside the evaluation call.
      steps: Number of steps for which to evaluate model. If `None`, evaluates
        until evaluation datasets raises an end-of-input exception.
      every_n_iter: `int`, runs the evaluator once every N training iteration.
      eval_dir: a folder to store the evaluation results

    Raises:
      ValueError: if `every_n_iter` is non-positive or it's not a single machine
        training
    '''
    if every_n_iter is None or every_n_iter <= 0:
      raise ValueError(f"invalid every_n_iter={every_n_iter}.")
    self._update_op = update_op
    self._eval_dict = eval_dict
    self._hooks = Context.get().evaluation_hooks
    if hooks is not None:
      self._hooks.extend(hooks)
    self._steps = steps
    self._every_n_iter = every_n_iter
    self._eval_dir = eval_dir
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_iter)

  def _clean_collection(self):
    r'''Prevent invocation of global variables.
    '''
    all_keys = ops.get_all_collection_keys()
    allowed_keys = [
        ops.GraphKeys.GLOBAL_VARIABLES,
        ops.GraphKeys.GLOBAL_STEP,
        ops.GraphKeys.LOCAL_VARIABLES]
    keys_to_clean = []
    for k in all_keys:
      if k not in allowed_keys:
        keys_to_clean.append(k)
    if keys_to_clean:
      self._retained_collection = \
          {k: ops.get_collection(k) for k in keys_to_clean}
      for k in keys_to_clean:
        ops.get_collection_ref(k).clear()
    else:
      self._retained_collection = None

  def _restore_collection(self):
    r'''Restore global variables.
    '''
    if self._retained_collection is not None:
      for k, v in self._retained_collection.items():
        ops.get_collection_ref(k).extend(v)

  def begin(self):
    r'''Preprocess global step and evaluation's hooks.
    '''
    self._timer.reset()
    self._iter_count = 0

    if ops.GraphKeys.GLOBAL_STEP not in self._eval_dict:
      global_step_tensor = training_util.get_global_step(
          ops.get_default_graph())
      self._eval_dict[ops.GraphKeys.GLOBAL_STEP] = global_step_tensor
    for h in self._hooks:
      h.begin()

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    r'''Call evaluation's hooks.
    '''
    if ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS):
      raise ValueError(
          'InMemoryEvaluator does not support saveables other than global '
          'variables.')
    for h in self._hooks:
      h.after_create_session(session, coord)

  def _call_eval_hook_before_run(
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
    eval_feed_dict = self._call_eval_hook_before_run(
        run_context, actual_fetches)
    eval_hook_res = run_context.session.run(
        actual_fetches, feed_dict=eval_feed_dict)
    for hook in self._hooks:
      hook.after_run(
          run_context,
          session_run_hook.SessionRunValues(
              results=eval_hook_res.pop(hook, None),
              options=config_pb2.RunOptions(),
              run_metadata=config_pb2.RunMetadata()))
    return eval_hook_res

  def _dict_to_str(self, dictionary):
    r'''Get a `str` representation of a `dict`.

    Args:
      dictionary: The `dict` to be represented as `str`.

    Returns:
      A `str` representing the `dictionary`.
    '''
    return ', '.join(f"{k} = {v}"
                     for k, v in sorted(six.iteritems(dictionary))
                     if not isinstance(v, six.binary_type))

  def _write_dict_to_summary(self, dictionary):
    r'''Write evaluation results to eval_dir.
    '''
    current_global_step = dictionary[ops.GraphKeys.GLOBAL_STEP]
    logging.info(
        'Saving dict for global step %d: %s',
        current_global_step, self._dict_to_str(dictionary))

    summary_writer = core_summary.FileWriterCache.get(self._eval_dir)
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
            summ.value[i].tag = f"{key}/{i}"
          summary_proto.value.extend(summ.value)
        except message.DecodeError:
          logging.warn(
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
        logging.warn(
            'Skipping summary for %s, must be a float, np.float32, np.int64, '
            'np.int32 or int or np.ndarray or a serialized string of Summary.',
            key)
    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()

  def _evaluate(self, run_context):
    for _ in range(self._steps):
      if not run_context.stop_requested:
        self._run(run_context, self._update_op)
    eval_dict_values = self._run(run_context, self._eval_dict)
    if eval_dict_values is not None:
      self._write_dict_to_summary(eval_dict_values)
    self._timer.update_last_triggered_step(self._iter_count)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    r'''Runs evaluator after session run.
    '''
    self._iter_count += 1
    if self._timer.should_trigger_for_step(self._iter_count):
      ctx_stop_requested = run_context.stop_requested
      run_context._stop_requested = False # pylint: disable=protected-access
      self._evaluate(run_context)
      run_context._stop_requested = ctx_stop_requested # pylint: disable=protected-access

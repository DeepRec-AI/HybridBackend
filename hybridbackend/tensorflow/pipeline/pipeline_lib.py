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

r'''A stack of pipelined layers that applies advanced functionality of
HybridBackend such as Gradient Aggregation.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order,unused-import
import re

# Base objects.
from tensorflow._api.v1 import train as train_v1
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import training as _training
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys
from hybridbackend.tensorflow.framework.ops import ModeKeys
from hybridbackend.tensorflow.framework.ops import MultiValues
from hybridbackend.tensorflow.framework.rewriting import GraphRewriting
from hybridbackend.tensorflow.training.variables import reuse_variables


class PipelinedValues(object):  # pylint: disable=useless-object-inheritance
  r'''Pipelined Values
  '''
  def __init__(self, name):
    r'''Creates a PipelinedValues
    '''
    self._name = name
    self._values = {}

  @property
  def name(self):
    return self._name

  @property
  def values(self):
    return self._values

  def add(self, key, value):
    if not isinstance(value, MultiValues):
      raise ValueError(f'MultiValues is expected for {key}')
    self._values[key] = value

  def get(self, key):
    if key not in self._values:
      raise ValueError(f'{key} not found in PipelinedValues')
    return self._values[key]


def wraps_pipelined_optimizer(cls):
  r'''Wraps optimizer to support pipelined training.
  '''
  class PipelinedOptimizer(cls):
    r'''Class to apply pipeline in optimizer.
    '''
    def _accumulate_grads(self, grads_splits):
      r'''Accumulate and compute gradients.

      Args:
        grads_splits: List of `Tensor` or list of list of
          tensors the same size as `ys` and holding the gradients
          computed for each y in `ys`.

      Returns:
        Accumulated gradients.
      '''
      if not grads_splits or grads_splits[0] is None:
        return None
      if isinstance(grads_splits[0], ops.IndexedSlices):
        tensor_values, tensor_indices = zip(
          *[(t.values, t.indices) for t in grads_splits])
        dense_shape = grads_splits[0].dense_shape
        accumulated_values = array_ops.concat(tensor_values, axis=0)
        accumulated_indices = array_ops.concat(tensor_indices, axis=0)
        return ops.IndexedSlices(
          accumulated_values, accumulated_indices, dense_shape)
      if isinstance(grads_splits[0], ops.Tensor):
        return math_ops.accumulate_n(grads_splits)
      raise ValueError(f'{grads_splits} are not all Tensors')

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=optimizer_lib.Optimizer.GATE_OP,
                 aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
      r'''Pipelined version of minimizing loss
      '''
      if Context.get().options.data_batch_count < 2:
        return super().minimize(
          loss, global_step=global_step, var_list=var_list,
          gate_gradients=gate_gradients, aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops, name=name,
          grad_loss=grad_loss)

      pipelined_values = ops.get_collection(GraphKeys.PIPLINED_VALUES)
      if (not pipelined_values) or\
         (not isinstance(pipelined_values[0], PipelinedValues)):
        raise ValueError('No pipelined values are found in collection')
      pipelined_loss = pipelined_values[0].get(loss.__repr__())
      grads_and_vars_pipelined = {}
      sharded_vars = ops.get_default_graph().get_collection_ref(
        GraphKeys.SHARDED_VARIABLES)

      for _, loss_val in enumerate(list(pipelined_loss.values)):
        grads_and_vars = self.compute_gradients(
          loss_val, var_list=var_list, gate_gradients=gate_gradients,
          aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops,
          grad_loss=grad_loss)
        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
          raise ValueError(
            'No gradients provided for any variable, check your graph for ops'
            ' that do not support gradients, between variables '
            f'{[str(v) for _, v in grads_and_vars]} and loss {loss_val}.')
        for g, v in grads_and_vars:
          if v in grads_and_vars_pipelined:
            grads_and_vars_pipelined[v].append(g)
          else:
            grads_and_vars_pipelined[v] = [g]
      grads_and_vars_output = []
      for v, grads in grads_and_vars_pipelined.items():
        if v in sharded_vars:
          for g in grads:
            grads_and_vars_output.append((g, v))
        else:
          grads_and_vars_output.append(
            (self._accumulate_grads(grads), v))
      return self.apply_gradients(
        grads_and_vars_output, global_step=global_step, name=name)

  return PipelinedOptimizer


class PipelinedOptimizerRewriting(GraphRewriting):
  r'''Rewriting optimizers for pipelined training.
  '''
  def __init__(self):
    super().__init__()
    self._prev_optimizers = {}

  def begin(self):
    r'''Rewrites API.
    '''
    for k, c in _training.__dict__.items():
      if (isinstance(c, type)
          and issubclass(c, _training.Optimizer)
          and c not in (
            _training.Optimizer,
            _training.SyncReplicasOptimizer)):
        self._prev_optimizers[k] = c
        wrapped = wraps_pipelined_optimizer(c)
        setattr(_training, k, wrapped)
        setattr(train_v1, k, wrapped)

  def end(self):
    r'''Revert API rewriting.
    '''
    for c, opt in self._prev_optimizers.items():
      setattr(_training, c, opt)
      setattr(train_v1, c, opt)


GraphRewriting.register(PipelinedOptimizerRewriting)


def _process_fn_args(fn, *args):
  r'''Process to eliminate first arg of pipeline function.
  '''
  if fn.__name__ == fn.__qualname__ or\
      '<locals>' in fn.__qualname__ or\
      re.sub(r'(#.*|\s)', '', tf_inspect.getsourcelines(fn)[0][0])\
      == '@staticmethod':
    return fn, args
  args_actual = args[1:]

  def fn_actual(*args_actual, **kwargs):
    return fn(args[0], *args_actual, **kwargs)
  return fn_actual, args_actual


def _call_pipeline(fn, *args, **kwargs):
  r''' invoke the pipeline
  '''
  args_flat = nest.flatten(args)
  args_unpack = []
  for item in args_flat:
    if not isinstance(item, MultiValues):
      raise ValueError('MultiValues is expected')
    args_unpack.append(list(item.values))
  args_unpack = list(zip(*args_unpack))
  args_processed = [nest.pack_sequence_as(args, item) for item in args_unpack]
  outputs = []
  with reuse_variables() as reuse_vars:
    kwargs_list = [kwargs] * Context.get().options.data_batch_count
    for input_id, input_val in enumerate(args_processed):
      if input_id > 0:
        reuse_vars(variable_scope.AUTO_REUSE)
      if not isinstance(input_val, tuple):
        input_val = (input_val,)
      outputs.append(fn(*input_val, **kwargs_list.pop(0)))
  pipelined_values = ops.get_collection_ref(GraphKeys.PIPLINED_VALUES)
  if not pipelined_values:
    pipelined_values.append(PipelinedValues(GraphKeys.PIPLINED_VALUES))
  pipelined_values[0].add(outputs[0].__repr__(), MultiValues.build(outputs))
  return outputs[0]


def compute_pipeline():
  r'''Decorator to pipeline training'''
  def decorated(fn):
    def wrapped_fn(*args, **kwargs):
      r'''Wrapped function.
      '''
      if (Context.get().options.data_batch_count > 1
          and Context.get().options.mode == ModeKeys.TRAIN):
        fn_actual, args_actual = _process_fn_args(fn, *args)
        return _call_pipeline(fn_actual, *args_actual, **kwargs)
      return fn(*args, **kwargs)
    return wrapped_fn
  return decorated

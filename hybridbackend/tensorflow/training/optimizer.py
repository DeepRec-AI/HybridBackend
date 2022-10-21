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

r'''Support for synchronous training using hybrid backends.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow._api.v1 import train as train_v1
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.training import training

from hybridbackend.tensorflow.distribute.gradient import aggregate_gradients
from hybridbackend.tensorflow.distribute.pubsub import PubSub
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys
from hybridbackend.tensorflow.framework.ops import ModeKeys
from hybridbackend.tensorflow.framework.rewriting import GraphRewriting
from hybridbackend.tensorflow.framework.rewriting import SessionRunRewriting


class HybridBackendOptimizerBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of optimizer wrapper.
  '''


def wraps_optimizer(
    cls,
    num_buckets=None,
    aggregation=None):
  r'''Decorator to create hybridbackend optimizer class.

  Args:
    optimizer_type: The actual optimizer type that will be used to compute and
      apply the gradients. Must be one of the Optimizer classes.
    num_buckets: Max number of gradient groups.
    aggregation: Aggregate gradients inside `compute_gradients` or
      `apply_gradients`.

  Returns:
    hb_optimizer_type: The hybridbackend optimizer type for `optimizer_type`
  '''
  if num_buckets is None:
    num_buckets = Context.get().options.grad_nbuckets
  if aggregation is None:
    aggregation = (
      'apply_gradients'
      if Context.get().options.grad_lazy_sync else 'compute_gradients')
  if issubclass(cls, HybridBackendOptimizerBase):
    return cls

  class HybridBackendOptimizer(cls, HybridBackendOptimizerBase):
    r'''Class to sync and aggregate gradients or the optimizer efficiently.
    '''
    def __init__(self, *args, **kwargs):
      r'''Constructs a hybrid backend optimizer.

      Args:
        *args: Arguments for compute_gradients().
        **kwargs: Keyword arguments for compute_gradients().
      '''
      super().__init__(*args, **kwargs)
      self._replica_vars = []
      self._shard_vars = []
      ctx = kwargs.pop('ctx', None)
      if not ctx:
        ctx = Context.get()
      if len(ctx.local_devices) > 1:
        raise NotImplementedError(
          'Multiple devices in one graph is not supported yet.')
      self._ctx = ctx

    def _create_slots(self, var_list):
      r'''Create all slots needed by the variables.

      Args:
        var_list: A list of `Variable` objects.
      '''
      super()._create_slots(var_list)
      sharded_vars = ops.get_default_graph().get_collection_ref(
        GraphKeys.SHARDED_VARIABLES)
      slot_names = self.get_slot_names()
      for v in sharded_vars:
        if v is None or not v.trainable:
          continue
        for slot_name in slot_names:
          ops.add_to_collection(
            GraphKeys.SHARDED_VARIABLES,
            self.get_slot(v, slot_name))

    def _aggregate_gradients(self, grads_and_vars):
      r'''Aggregate gradients to variables.

      Args:
        grads_and_vars: List of (gradient, variable) pairs as returned by
          compute_gradients().

      Returns:
        Aggregated gradients to variables.
      '''
      if distribution_strategy_context.has_distribution_strategy():
        # Do nothing if this function resides in a distribution context.
        return grads_and_vars

      return aggregate_gradients(grads_and_vars, num_buckets)

    def compute_gradients(self, *args, **kwargs):
      r'''Compute gradients of "loss" for the variables in "var_list".

      This simply wraps the compute_gradients() from the real optimizer. The
      gradients will be aggregated in the apply_gradients() so that user can
      modify the gradients like clipping with per replica global norm if needed.
      The global norm with aggregated gradients can be bad as one replica's huge
      gradients can hurt the gradients from other replicas.

      Args:
        *args: Arguments for compute_gradients().
        **kwargs: Keyword arguments for compute_gradients().

      Returns:
        A list of (gradient, variable) pairs.
      '''
      grads_and_vars = super().compute_gradients(*args, **kwargs)
      if aggregation == 'compute_gradients':
        grads_and_vars = self._aggregate_gradients(grads_and_vars)
      return grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
      r'''Apply gradients to variables.

      This contains most of the synchronization implementation and also wraps
      the apply_gradients() from the real optimizer.

      Args:
        grads_and_vars: List of (gradient, variable) pairs as returned by
          compute_gradients().
        global_step: Optional Variable to increment by one after the
          variables have been updated.
        name: Optional name for the returned operation.  Default to the
          name passed to the Optimizer constructor.

      Returns:
        train_op: The op to dequeue a token so the replicas can exit this batch
          and start the next one. This is executed by each replica.

      Raises:
        ValueError: If the grads_and_vars is empty.
        ValueError: If global step is not provided, the staleness cannot be
          checked.
      '''
      if aggregation != 'compute_gradients':
        grads_and_vars = self._aggregate_gradients(grads_and_vars)
      _, self._variables = zip(*grads_and_vars)
      return super().apply_gradients(grads_and_vars, global_step, name=name)

  return HybridBackendOptimizer


class OptimizerRewriting(GraphRewriting):
  r'''Rewriting optimizers.
  '''
  def __init__(self):
    super().__init__()
    self._prev_optimizers = {}

  def begin(self):
    r'''Rewrites API.
    '''
    for k, c in training.__dict__.items():
      if (isinstance(c, type)
          and issubclass(c, training.Optimizer)
          and c not in (
            training.Optimizer,
            training.SyncReplicasOptimizer)):
        self._prev_optimizers[k] = c
        wrapped = wraps_optimizer(c)
        setattr(training, k, wrapped)
        setattr(train_v1, k, wrapped)

  def end(self):
    r'''Revert API rewriting.
    '''
    for c, opt in self._prev_optimizers.items():
      setattr(training, c, opt)
      setattr(train_v1, c, opt)


GraphRewriting.register(OptimizerRewriting)


class VariablesInitializationRewriting(SessionRunRewriting):
  r'''A SessionRunHook initializes variables across devices.
  '''
  def begin(self):
    r''' initialize replica variables and enable synchronous dataset wrapper
    '''
    with ops.device(Context.get().devices[Context.get().rank]):
      replicated_variables = list(
        collections.OrderedDict.fromkeys(
          ops.get_collection_ref(GraphKeys.TRAINABLE_REPLICATED)))
      for v in replicated_variables:
        try:
          self._set_initializer(v, Context.get().devices)
        except:  # pylint: disable=bare-except
          pass

  def _get_initial_value(self, var):
    r'''Get initial value of a variable without uninitialized dependencies.

    NOTE: `_try_guard_against_uninitialized_dependencies` is no longer a
          method of a variable since tensorflow 1.15
    '''
    # pylint:disable=protected-access
    if hasattr(var, '_try_guard_against_uninitialized_dependencies'):
      return var._try_guard_against_uninitialized_dependencies(
        var._initial_value)
    return variables._try_guard_against_uninitialized_dependencies(
      var.name, var._initial_value)
    # pylint:enable=protected-access

  def _get_initializer_op(self, var):
    r'''Get initializer op of any kind of variable.
    '''
    # pylint:disable=protected-access
    if isinstance(var, resource_variable_ops.ResourceVariable):
      return gen_resource_variable_ops.assign_variable_op(
        var._handle, self._get_initial_value(var))
    return state_ops.assign(var._variable, self._get_initial_value(var)).op
    # pylint:enable=protected-access

  def _set_initializer(self, var, devices, name=None):
    r'''Initialize variables.
    '''
    if name is None:
      name = var.name.split(':')[0]
    rank = Context.get().current_index()
    with ops.name_scope('initializers/'):
      with ops.name_scope(f'{name}/initializer'):
        with ops.control_dependencies(None):
          with ops.device(var.device):
            initial_value = var.initial_value
            if callable(initial_value):
              initial_value = initial_value()
            initial_value = array_ops.identity(
              ops.convert_to_tensor(initial_value))
            # pylint:disable=protected-access
            var._initial_value = initial_value
            if len(devices) > 1:
              var._initial_value = PubSub(devices, rank=rank)(
                lambda: initial_value,
                initial_value.shape,
                initial_value.dtype,
                name=f'{name}_pubsub')
            var._initializer_op = self._get_initializer_op(var)
            # pylint:enable=protected-access


SessionRunRewriting.register(
  VariablesInitializationRewriting, [ModeKeys.TRAIN])

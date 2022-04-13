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

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.training import session_run_hook

from hybridbackend.tensorflow.distribute.pubsub import PubSub
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys
from hybridbackend.tensorflow.training.optimizer_lib import GradientAggregation
from hybridbackend.tensorflow.training.saver import replace_default_saver


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
    num_buckets = Context.get().param(
      'optimizer_nbuckets', 1,
      env='HB_OPTIMIZER_NBUCKETS',
      parser=int)
  if aggregation is None:
    aggregation = Context.get().param(
      'optimizer_aggregation', 'apply_gradients',
      env='HB_OPTIMIZER_AGGREGATION')
    if aggregation not in ('compute_gradients', 'apply_gradients'):
      raise ValueError(
        'aggregation must be one of `compute_gradients`, `apply_gradients`')

  class HybridBackendOptimizer(cls):
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

      if len(self._ctx.devices) <= 1:
        return grads_and_vars

      shard_grads = []
      shard_grad_indices = []
      replica_grads = []
      replica_grad_indices = []
      sharded_vars = ops.get_default_graph().get_collection_ref(
        GraphKeys.SHARDED_VARIABLES)
      sharded_vars += ops.get_default_graph().get_collection_ref(
        GraphKeys.NOT_REPLICATED)
      for i, gv in enumerate(grads_and_vars):
        if gv[0] is None:
          continue
        if gv[1] in sharded_vars:
          shard_grad_indices.append(i)
          shard_grads.append(gv[0])
          self._shard_vars.append(gv[1])
        else:
          replica_grad_indices.append(i)
          replica_grads.append(gv[0])
          self._replica_vars.append(gv[1])
      aggregate = GradientAggregation(self._ctx.devices, num_buckets)
      cross_replica_grads, shard_grads = aggregate(replica_grads, shard_grads)

      indexed_grads_and_vars = []
      for i, g in enumerate(shard_grads):
        indexed_grads_and_vars.append(
          (shard_grad_indices[i], (g, self._shard_vars[i])))
      for i, g in enumerate(cross_replica_grads):
        indexed_grads_and_vars.append(
          (replica_grad_indices[i], (g, self._replica_vars[i])))
      _, grads_and_vars = zip(*sorted(indexed_grads_and_vars))
      self._ctx.add_training_hook(
        HybridBackendOptimizerHook(
          self._replica_vars,
          self._shard_vars,
          self._ctx.devices,
          self._ctx.current_device()))
      return grads_and_vars

    def _accumulate_grads(self, grads_splits):
      r'''Accumulate and compute gradients.

      Args:
        grads_splits: List of `Tensor` or list of list of tensors the same size
          as `ys` and holding the gradients computed for each y in `ys`.

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

    def _aggregate_pipelined_gradients(self, grads_and_vars):
      grads_and_vars_processed = []
      for gv in grads_and_vars:
        if hasattr(gv[0], 'pipelined_grads'):
          grads_and_vars_processed.append(
            (self._accumulate_grads(gv[0].pipelined_grads), gv[1]))
        else:
          grads_and_vars_processed.append((gv[0], gv[1]))
      return self._aggregate_gradients(grads_and_vars_processed)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
      r'''See base class.
      '''
      with ops.colocate_with(grad, ignore_existing=True):
        unique_indices, new_index_positions = array_ops.unique(indices)
        summed_grad = math_ops.unsorted_segment_sum(
          grad, new_index_positions,
          array_ops.shape(unique_indices)[0])
      try:
        apply_op = self._resource_apply_sparse(
          summed_grad, handle, unique_indices)
      except NotImplementedError:
        apply_op = super()._resource_apply_sparse_duplicate_indices(  # pylint: disable=protected-access
          grad, handle, indices)
      return apply_op

    def _apply_sparse_duplicate_indices(self, grad, var):
      r'''See base class.
      '''
      with ops.colocate_with(grad, ignore_existing=True):
        unique_indices, new_index_positions = array_ops.unique(grad.indices)
        summed_values = math_ops.unsorted_segment_sum(
          grad.values, new_index_positions,
          array_ops.shape(unique_indices)[0])
        gradient_no_duplicate_indices = ops.IndexedSlices(
          indices=unique_indices,
          values=summed_values,
          dense_shape=grad.dense_shape)
      try:
        apply_op = self._apply_sparse(gradient_no_duplicate_indices, var)
      except NotImplementedError:
        apply_op = super()._apply_sparse_duplicate_indices(grad, var)  # pylint: disable=protected-access
      return apply_op

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

    def make_session_run_hook(self):
      r'''Creates a hook to handle hook ops such as initialization.
      '''
      return HybridBackendOptimizerHook(
        self._replica_vars,
        self._shard_vars,
        self._ctx.devices,
        self._ctx.current_device())
  return HybridBackendOptimizer


class HybridBackendOptimizerHook(session_run_hook.SessionRunHook):
  r'''A SessionRunHook initializes variables across devices.
  '''
  def __init__(self, replica_vars, shard_vars, devices, device):
    r'''Creates hook to initialize variables across devices.

    Args:
      replica_vars: Replications of variables.
      shard_vars: Shards of variables.
      devices: Devices involved.
      device: Optimizer device.
    '''
    super().__init__()
    self._replica_vars = replica_vars
    self._shard_vars = shard_vars
    self._devices = devices
    self._device = device

  def begin(self):
    r''' initialize replica variables and enable synchronous dataset wrapper
    '''
    with ops.device(self._device):
      for v in self._replica_vars:
        self._set_initializer(v)
    replace_default_saver()

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

  def _set_initializer(self, var, name=None):
    r'''Initialize variables.
    '''
    if name is None:
      name = var.name.split(':')[0]
    rank = Context.get().current_index()
    with ops.name_scope('initializers'):
      with ops.name_scope(name):
        with ops.control_dependencies(None):
          with ops.device(var.device):
            initial_value = var.initial_value
            if callable(initial_value):
              initial_value = initial_value()
            initial_value = array_ops.identity(
              ops.convert_to_tensor(initial_value))
            # pylint:disable=protected-access
            var._initial_value = initial_value
            if len(self._devices) > 1:
              var._initial_value = PubSub(self._devices, rank=rank)(
                lambda: initial_value,
                initial_value.shape,
                initial_value.dtype,
                name=f'{name}_initializer')
            var._initializer_op = self._get_initializer_op(var)
            # pylint:enable=protected-access

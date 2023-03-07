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
import time

from tensorflow._api.v1 import train as train_v1
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training

from hybridbackend.tensorflow.distribute.rpc import RpcCollective
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys
from hybridbackend.tensorflow.framework.ops import ModeKeys
from hybridbackend.tensorflow.framework.rewriting import GraphRewriting
from hybridbackend.tensorflow.framework.rewriting import SessionRunRewriting
from hybridbackend.tensorflow.training.gradient import aggregate_gradients


class HybridBackendOptimizerBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of optimizer wrapper.
  '''


def wraps_optimizer(
    cls,
    aggregation=None):
  r'''Decorator to create hybridbackend optimizer class.

  Args:
    optimizer_type: The actual optimizer type that will be used to compute and
      apply the gradients. Must be one of the Optimizer classes.
    aggregation: Aggregate gradients inside `compute_gradients` or
      `apply_gradients`.

  Returns:
    hb_optimizer_type: The hybridbackend optimizer type for `optimizer_type`
  '''
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

      return aggregate_gradients(grads_and_vars)

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
          self._set_initializer(v)
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

  def _set_initializer(self, var, name=None):
    r'''Initialize variables.
    '''
    if name is None:
      name = var.name.split(':')[0]
    devices = Context.get().devices
    rank = Context.get().rank
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
              var._initial_value = (
                RpcCollective(devices, rank).broadcast(
                  lambda: initial_value,
                  initial_value.dtype,
                  initial_value.shape,
                  name=f'{name}_initial_value/rpc_broadcast'))
            var._initializer_op = self._get_initializer_op(var)
            # pylint:enable=protected-access


SessionRunRewriting.register(
  VariablesInitializationRewriting, [ModeKeys.TRAIN])


class SyncReplicasOptimizer(optimizer.Optimizer):
  r'''Class to synchronize, aggregate gradients and pass them to the optimizer.

  In a typical asynchronous training environment, it's common to have some
  stale gradients. For example, with a N-replica asynchronous training,
  gradients will be applied to the variables N times independently. Depending
  on each replica's training speed, some gradients might be calculated from
  copies of the variable from several steps back (N-1 steps on average). This
  optimizer avoids stale gradients by collecting gradients from all replicas,
  averaging them, then applying them to the variables in one shot, after
  which replicas can fetch the new variables and continue.

  The following accumulators/queue are created:

  * N `gradient accumulators`, one per variable to train. Gradients are pushed
    to them and the chief worker will wait until enough gradients are collected
    and then average them before applying to variables. The accumulator will
    drop all stale gradients (more details in the accumulator op).
  * 1 `token` queue where the optimizer pushes the new global_step value after
    all variables are updated.

  The following local variable is created:
  * `sync_rep_local_step`, one per replica. Compared against the global_step in
    each accumulator to check for staleness of the gradients.

  The optimizer adds nodes to the graph to collect gradients and pause the
  trainers until variables are updated.
  For the Parameter Server job:

  1. An accumulator is created for each variable, and each replica pushes the
     gradients into the accumulators instead of directly applying them to the
     variables.
  2. Each accumulator averages once enough gradients (replicas_to_aggregate)
     have been accumulated.
  3. Apply the averaged gradients to the variables.
  4. Only after all variables have been updated, increment the global step.
  5. Only after step 4, pushes `global_step` in the `token_queue`, once for
     each worker replica. The workers can now fetch the global step, use it to
     update its local_step variable and start the next batch. Please note that
     some workers can consume multiple minibatches, while some may not consume
     even one. This is because each worker fetches minibatches as long as
     a token exists. If one worker is stuck for some reason and does not
     consume a token, another worker can use it.

  For the replicas:

  1. Start a step: fetch variables and compute gradients.
  2. Once the gradients have been computed, push them into gradient
     accumulators. Each accumulator will check the staleness and drop the stale.
  3. After pushing all the gradients, dequeue an updated value of global_step
     from the token queue and record that step to its local_step variable. Note
     that this is effectively a barrier.
  4. Start the next batch.

  ### Usage

  ```python
  # Create any optimizer to update the variables, say a simple SGD:
  opt = GradientDescentOptimizer(learning_rate=0.1)

  # Wrap the optimizer with sync_replicas_optimizer with 50 replicas: at each
  # step the optimizer collects 50 gradients before applying to variables.
  # Note that if you want to have 2 backup replicas, you can change
  # total_num_replicas=52 and make sure this number matches how many physical
  # replicas you started in your job.
  opt = tf.compat.v1.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=50,
                                 total_num_replicas=50)

  # Some models have startup_delays to help stabilize the model but when using
  # sync_replicas training, set it to 0.

  # Now you can call `minimize()` or `compute_gradients()` and
  # `apply_gradients()` normally
  training_op = opt.minimize(total_loss, global_step=self.global_step)


  # You can create the hook which handles initialization and queues.
  sync_replicas_hook = opt.make_session_run_hook(is_chief)
  ```

  In the training program, every worker will run the train_op as if not
  synchronized.

  ```python
  with training.MonitoredTrainingSession(
      master=workers[worker_id].target, is_chief=is_chief,
      hooks=[sync_replicas_hook]) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(training_op)
  ```

  To use SyncReplicasOptimizer with an `Estimator`, you need to send
  sync_replicas_hook while calling the fit.
  ```python
  my_estimator = DNNClassifier(..., optimizer=opt)
  my_estimator.fit(..., hooks=[sync_replicas_hook])
  ```
  '''

  def __init__(self,
               opt,
               replicas_to_aggregate,
               total_num_replicas=None,
               variable_averages=None,
               variables_to_average=None,
               use_locking=False,
               name='sync_replicas'):
    r'''Construct a sync_replicas optimizer.

    Args:
      opt: The actual optimizer that will be used to compute and apply the
        gradients. Must be one of the Optimizer classes.
      replicas_to_aggregate: number of replicas to aggregate for each variable
        update.
      total_num_replicas: Total number of tasks/workers/replicas, could be
        different from replicas_to_aggregate.
        If total_num_replicas > replicas_to_aggregate: it is backup_replicas +
        replicas_to_aggregate.
        If total_num_replicas < replicas_to_aggregate: Replicas compute
        multiple batches per update to variables.
      variable_averages: Optional `ExponentialMovingAverage` object, used to
        maintain moving averages for the variables passed in
        `variables_to_average`.
      variables_to_average: a list of variables that need to be averaged. Only
        needed if variable_averages is passed in.
      use_locking: If True use locks for update operation.
      name: string. Optional name of the returned operation.
    '''
    if total_num_replicas is None:
      total_num_replicas = replicas_to_aggregate

    super().__init__(use_locking, name)
    logging.info(
      f'SyncReplicasV3: replicas_to_aggregate={replicas_to_aggregate}; '
      f'total_num_replicas={total_num_replicas}')
    self._opt = opt
    self._replicas_to_aggregate = replicas_to_aggregate
    self._gradients_applied = False
    self._variable_averages = variable_averages
    self._variables_to_average = variables_to_average
    self._total_num_replicas = total_num_replicas
    self._tokens_per_step = max(total_num_replicas, replicas_to_aggregate)
    self._global_step = None
    self._sync_token_queue = None

    # The synchronization op will be executed in a queue runner which should
    # only be executed by one of the replicas (usually the chief).
    self._chief_queue_runner = None

    # Remember which accumulator is on which device to set the initial step in
    # the accumulator to be global step. This list contains list of the
    # following format: (accumulator, device).
    self._accumulator_list = []

  def compute_gradients(self, *args, **kwargs):
    r'''Compute gradients of 'loss' for the variables in 'var_list'.

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
    return self._opt.compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    r'''Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer.

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
    del name

    if not grads_and_vars:
      raise ValueError('Must supply at least one variable')

    if global_step is None:
      raise ValueError('Global step is required to check staleness')

    self._global_step = global_step
    train_ops = []
    aggregated_grad = []
    var_list = []

    # local_anchor op will be placed on this worker task by default.
    local_anchor = control_flow_ops.no_op()
    # Colocating local_step variable prevents it being placed on the PS.
    distribution_strategy = distribution_strategy_context.get_strategy()
    with distribution_strategy.extended.colocate_vars_with(local_anchor):
      self._local_step = variable_scope.variable(
        initial_value=0,
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        dtype=global_step.dtype.base_dtype,
        name='sync_rep_local_step')

    self.local_step_init_op = state_ops.assign(self._local_step, global_step)
    chief_init_ops = [self.local_step_init_op]
    self.ready_for_local_init_op = variables.report_uninitialized_variables(
      variables.global_variables())

    with ops.name_scope(None, self._name):
      for grad, var in grads_and_vars:
        var_list.append(var)
        with ops.device(var.device):
          # Dense gradients.
          if grad is None:
            aggregated_grad.append(None)  # pass-through.
            continue
          if isinstance(grad, ops.Tensor):
            grad_accum = data_flow_ops.ConditionalAccumulator(
              grad.dtype,
              shape=var.get_shape(),
              shared_name=var.name + '/grad_accum')
            train_ops.append(grad_accum.apply_grad(
              grad, local_step=self._local_step))
            aggregated_grad.append(grad_accum.take_grad(
              self._replicas_to_aggregate))
          else:
            if not isinstance(grad, ops.IndexedSlices):
              raise ValueError('Unknown grad type!')
            grad_accum = data_flow_ops.SparseConditionalAccumulator(
              grad.dtype, shape=(), shared_name=var.name + '/grad_accum')
            train_ops.append(grad_accum.apply_indexed_slices_grad(
              grad, local_step=self._local_step))
            aggregated_grad.append(grad_accum.take_indexed_slices_grad(
              self._replicas_to_aggregate))

          self._accumulator_list.append((grad_accum, var.device))

      aggregated_grads_and_vars = zip(aggregated_grad, var_list)

      # sync_op will be assigned to the same device as the global step.
      with ops.device(global_step.device), ops.name_scope(''):
        update_op = self._opt.apply_gradients(aggregated_grads_and_vars,
                                              global_step)

      # Create token queue.
      with ops.device(global_step.device), ops.name_scope(''):
        sync_token_queue = (
          data_flow_ops.FIFOQueue(-1,
                                  global_step.dtype.base_dtype,
                                  shapes=(),
                                  name='sync_token_q',
                                  shared_name='sync_token_q'))
        self._sync_token_queue = sync_token_queue

        # dummy_queue is passed to the queue runner. Don't use the real queues
        # because the queue runner doesn't automatically reopen it once it
        # closed queues in PS devices.
        dummy_queue = (
          data_flow_ops.FIFOQueue(1,
                                  types_pb2.DT_INT32,
                                  shapes=(),
                                  name='dummy_queue',
                                  shared_name='dummy_queue'))

      with ops.device(global_step.device), ops.name_scope(''):
        # Replicas have to wait until they can get a token from the token queue.
        with ops.control_dependencies(train_ops):
          token = sync_token_queue.dequeue()
        train_op = state_ops.assign(self._local_step, token)

        self._remained_tokens = variable_scope.variable(
          initial_value=self._tokens_per_step,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          dtype=dtypes.int32,
          name='sync_rep_remained_tokens')

        with ops.control_dependencies([update_op]):
          # Sync_op needs to insert tokens to the token queue at the end of the
          # step so the replicas can fetch them to start the next step.
          tokens = array_ops.fill([self._tokens_per_step], global_step)
          sync_op = sync_token_queue.enqueue_many((tokens,))

        if self._variable_averages is not None:
          with ops.control_dependencies([sync_op]), ops.name_scope(''):
            sync_op = self._variable_averages.apply(
              self._variables_to_average)

        self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                            [sync_op])

        self._sync_at_end_op = control_flow_ops.group([
          sync_token_queue.enqueue((self._local_step,)),
          self._remained_tokens.assign_sub(1)])

      for accum, dev in self._accumulator_list:
        with ops.device(dev):
          chief_init_ops.append(
            accum.set_global_step(
              global_step, name='SetGlobalStep'))
      self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
      self._gradients_applied = True
      return train_op

  def get_chief_queue_runner(self):
    r'''Returns the QueueRunner for the chief to execute.

    This includes the operations to synchronize replicas: aggregate gradients,
    apply to variables, increment global step, insert tokens to token queue.

    Note that this can only be called after calling apply_gradients() which
    actually generates this queuerunner.

    Returns:
      A `QueueRunner` for chief to execute.

    Raises:
      ValueError: If this is called before apply_gradients().
    '''
    if self._gradients_applied is False:
      raise ValueError('Should be called after apply_gradients().')

    return self._chief_queue_runner

  def get_slot(self, *args, **kwargs):
    r'''Return a slot named 'name' created for 'var' by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    '''
    return self._opt.get_slot(*args, **kwargs)

  def variables(self):
    r'''Fetches a list of optimizer variables in the default graph.

    This wraps `variables()` from the actual optimizer. It does not include
    the `SyncReplicasOptimizer`'s local step.

    Returns:
      A list of variables.
    '''
    return self._opt.variables()

  def get_slot_names(self, *args, **kwargs):
    r'''Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    '''
    return self._opt.get_slot_names(*args, **kwargs)

  def get_init_tokens_op(self, num_tokens=-1):
    r'''Returns the op to fill the sync_token_queue with the tokens.

    This is supposed to be executed in the beginning of the chief/sync thread
    so that even if the total_num_replicas is less than replicas_to_aggregate,
    the model can still proceed as the replicas can compute multiple steps per
    variable update. Make sure:
    `num_tokens >= replicas_to_aggregate - total_num_replicas`.

    Args:
      num_tokens: Number of tokens to add to the queue.

    Returns:
      An op for the chief/sync replica to fill the token queue.

    Raises:
      ValueError: If this is called before apply_gradients().
      ValueError: If num_tokens are smaller than replicas_to_aggregate -
        total_num_replicas.
    '''
    if self._gradients_applied is False:
      raise ValueError(
        'get_init_tokens_op() should be called after apply_gradients().')

    tokens_needed = self._replicas_to_aggregate - self._total_num_replicas
    if num_tokens == -1:
      num_tokens = self._replicas_to_aggregate
    elif num_tokens < tokens_needed:
      raise ValueError(
        'Too few tokens to finish the first step: '
        f'{num_tokens} (given) vs {tokens_needed} (needed)')

    if num_tokens > 0:
      with ops.device(self._global_step.device), ops.name_scope(''):
        tokens = array_ops.fill([num_tokens], self._global_step)
        init_tokens = self._sync_token_queue.enqueue_many((tokens,))
    else:
      init_tokens = control_flow_ops.no_op(name='no_init_tokens')

    return init_tokens

  def sync_at_end(self, session, is_chief):
    r'''Synchronize at end of training.
    '''
    session.run(self._sync_at_end_op)
    if is_chief:
      while True:
        remained_tokens, step = session.run([
          self._remained_tokens, self._global_step])
        if remained_tokens > 1:
          logging.info(
            f'There are {remained_tokens} workers '
            f'still running at step: {step}')
          time.sleep(30)
        elif remained_tokens > 0:
          logging.info(f'There are 1 worker still running at step: {step}')
          time.sleep(10)
        else:
          logging.info(f'Training stopped at final step: {step}')
          break

  def make_session_run_hook(self, is_chief, num_tokens=-1):
    r'''Creates a hook to handle SyncReplicasHook ops such as initialization.'''
    return _SyncReplicasOptimizerHook(self, is_chief, num_tokens)


class _SyncReplicasOptimizerHook(session_run_hook.SessionRunHook):
  r'''A SessionRunHook handles ops related to SyncReplicasOptimizer.'''

  def __init__(self, sync_optimizer, is_chief, num_tokens):
    r'''Creates hook to handle SyncReplicasOptimizer initialization ops.

    Args:
      sync_optimizer: `SyncReplicasOptimizer` which this hook will initialize.
      is_chief: `Bool`, whether is this a chief replica or not.
      num_tokens: Number of tokens to add to the queue.
    '''
    self._sync_optimizer = sync_optimizer
    self._is_chief = is_chief
    self._num_tokens = num_tokens

  def begin(self):
    r'''Called once before using the session.
    '''
    if self._sync_optimizer._gradients_applied is False:  # pylint: disable=protected-access
      raise ValueError(
        'SyncReplicasOptimizer.apply_gradient should be called before using '
        'the hook.')
    if self._is_chief:
      self._local_init_op = self._sync_optimizer.chief_init_op
      self._ready_for_local_init_op = (
        self._sync_optimizer.ready_for_local_init_op)
      self._q_runner = self._sync_optimizer.get_chief_queue_runner()
      self._init_tokens_op = self._sync_optimizer.get_init_tokens_op(
        self._num_tokens)
    else:
      self._local_init_op = self._sync_optimizer.local_step_init_op
      self._ready_for_local_init_op = (
        self._sync_optimizer.ready_for_local_init_op)
      self._q_runner = None
      self._init_tokens_op = None

  def after_create_session(self, session, coord):
    r'''Runs SyncReplicasOptimizer initialization ops.'''
    local_init_success, msg = session_manager._ready(  # pylint: disable=protected-access
      self._ready_for_local_init_op, session,
      'Model is not ready for SyncReplicasOptimizer local init.')
    if not local_init_success:
      raise RuntimeError(
        'Init operations did not make model ready for SyncReplicasOptimizer '
        f'local_init. Init op: {self._local_init_op.name}, error: {msg}')
    session.run(self._local_init_op)
    if self._init_tokens_op is not None:
      session.run(self._init_tokens_op)
    if self._q_runner is not None:
      self._q_runner.create_threads(
        session, coord=coord, daemon=True, start=True)

  def end(self, session):
    self._sync_optimizer.sync_at_end(session, self._is_chief)

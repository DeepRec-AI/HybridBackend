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

r'''Classes for collective communication.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
try:
  from tensorflow.python.training import device_util
except: # pylint: disable=bare-except
  from tensorflow.python.distribute import device_util

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.distribute.communicator import Communicator


class CommunicatorPool(object):  # pylint: disable=useless-object-inheritance
  r'''Pool of communicators.
  '''
  _instances = {}

  @classmethod
  def get(cls, name=None):
    r'''Get singleton with specific name.

    Args:
      name: Name of the communicator pool.
    '''
    if name is None:
      name = Context.get().param(
          'comm_pool_name',
          default='default',
          env='HB_COMM_POOL_NAME')
    if name not in cls._instances:
      cls._instances[name] = cls()
    return cls._instances[name]

  def __init__(self, impl=None, capacity=None):
    r'''Initialize the communciator pool.

    Args:
      capacity: Capacity of the communciator pool.
    '''
    if capacity is None:
      capacity = Context.get().param(
          'comm_pool_capacity',
          default=1,
          env='HB_COMM_POOL_CAPACITY',
          parser=int)
    self._impl = Communicator.as_type(impl)
    self._capacity = self._impl.compute_pool_capacity(capacity)
    self._pool = []
    self._calls = 0
    self._deps = []

  @property
  def capacity(self):
    r'''Number of communicators in the pool.
    '''
    return self._capacity

  @property
  def size(self):
    r'''Size of communicators in the pool.
    '''
    return len(self._pool)

  def setup(self, impl=None, capacity=None):
    r'''Set up the communicator pool.
    Args:
      impl: Implementation of communicator.
      capacity: Number of communicators in the pool.
    '''
    if self._calls > 0:
      raise ValueError('Pool cannot be set up after use')
    self._impl = Communicator.as_type(impl)
    self._capacity = self._impl.compute_pool_capacity(capacity)

  def _build_communicator(self):
    r'''Create a new communicator.
    '''
    if 'GPU' not in Context.get().current_device().upper():
      local_gpus = [device_util.canonicalize(f'/gpu:{d}') \
          for d in xrange(Context.get().num_gpus)]
      with ops.device(f'/gpu:{Context.get().rank}'):
        return Communicator.create(
            ops.get_default_graph().get_name_scope(),
            local_gpus,
            impl=self._impl)
    return Communicator.create(
        ops.get_default_graph().get_name_scope(),
        Context.get().devices,
        impl=self._impl)

  def call(self, fn, inputs, trainable=True, immutable=True):
    r'''Call fn within communicator pool.

    Args:
      fn: Function accepts communicator, and returns results and grad function.
      inputs: Inputs of fn.
      trainable: If True, fn should return results and a function for gradients.
      immutable: If True, grad(fn) do not accepts variables as arguments.

    Returns:
      results: Result of fn.
    '''
    if not isinstance(inputs, (tuple, list)):
      inputs = [inputs]
    inputs = list(inputs)
    if self._capacity is None:
      comm = self._build_communicator()
      results, _ = fn(comm, inputs, None)
      self._calls += 1
      return results

    if len(self._pool) < self._capacity:
      self._pool.append(self._build_communicator())
      self._deps.append(control_flow_ops.no_op())

    if not trainable:
      comm_idx = self._calls % self._capacity
      comm = self._pool[comm_idx]
      results, _ = fn(comm, inputs, [self._deps[comm_idx]])
      self._deps[comm_idx] = control_flow_ops.group(results)
      self._calls += 1
      return results

    @custom_gradient.custom_gradient
    def wrapped_fn(*args):
      r'''Call with custom gradients.
      '''
      args = list(args)
      comm_idx = self._calls % self._capacity
      comm = self._pool[comm_idx]
      results, fn_grad = fn(comm, args, [self._deps[comm_idx]])
      self._deps[comm_idx] = control_flow_ops.group(results)
      self._calls += 1

      if immutable:
        def grad_fn(*grads):
          r'''Gradient function for fn.
          '''
          d_inputs, _ = fn_grad(list(grads), [self._deps[comm_idx]], None)
          self._deps[comm_idx] = control_flow_ops.group([
              g.values if isinstance(g, ops.IndexedSlices) else g
              for g in d_inputs if g is not None])
          return d_inputs
        return results, grad_fn

      def mutable_grad_fn(*grads, **kwargs):
        r'''Gradient function for fn.
        '''
        roots = list(kwargs.get('variables', []))
        d_inputs, d_roots = fn_grad(
            list(grads), [self._deps[comm_idx]], roots)
        self._deps[comm_idx] = control_flow_ops.group([
            g.values if isinstance(g, ops.IndexedSlices) else g
            for g in d_inputs + d_roots if g is not None])
        return d_inputs, d_roots
      return results, mutable_grad_fn
    return wrapped_fn(*inputs)

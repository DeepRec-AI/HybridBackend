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

r'''Classes for collective communication pooling.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import threading

from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient

try:
  from tensorflow.python.training import device_util
except:  # pylint: disable=bare-except
  from tensorflow.python.distribute import device_util

from hybridbackend.tensorflow.distribute.communicator import Communicator
from hybridbackend.tensorflow.framework.context import Context


class PubSub(object):  # pylint: disable=useless-object-inheritance
  r'''Publish/subscribe messages.
  '''
  _lock = threading.Lock()
  _thread_local = threading.local()

  _prev_channel = 1
  _devices_to_channel = {}

  _prev_call = 1000000
  _name_to_call = {}

  def __init__(self, devices, rank=None, root_rank=None):
    r'''Constructs a pub/sub instance.

    Args:
      devices: Involved devices.
      rank: Rank of this peer.
      root_rank: Rank of the root peer.
    '''
    self._bcast_device = Context.get().options.comm_pubsub_device
    self._devices = devices
    parsed_devices = [pydev.DeviceSpec.from_string(d) for d in devices]
    sorted_devices = sorted(
      [f'{d.device_type}:{d.device_index}' for d in parsed_devices])
    devices_str = ','.join(sorted_devices)
    with PubSub._lock:
      if devices_str not in PubSub._devices_to_channel:
        channel = PubSub._prev_channel
        PubSub._prev_channel += 1
        PubSub._devices_to_channel[devices_str] = channel
    self._channel = PubSub._devices_to_channel[devices_str]

    if rank is None:
      device = Context.current_device()
      if device not in self._devices:
        raise ValueError(
          f'Current device {device} not in devices {self._devices}')
      rank = self._devices.index(device)
    self._rank = rank
    if root_rank is None:
      root_rank = 0
    self._root_rank = root_rank

  @property
  def channel(self):
    r'''Channel for pub/sub.
    '''
    return self._channel

  def __call__(self, fn, shape, dtype, name=None):
    r'''Publish/subscribe message across devices.

    Args:
      fn: Function to generate tensor.
      shape: Shape of the generate tensor.
      dtype: Data type of the generated tensor.
      name: Name of the call.

    Returns:
      Published value.
    '''
    if len(self._devices) == 1:
      return fn()

    if name is None:
      name = ops.get_default_graph().unique_name('pubsub')
    name = name.replace(':', '_').replace('/', '_')

    with PubSub._lock:
      if name not in PubSub._name_to_call:
        call = PubSub._prev_call
        PubSub._prev_call += 1
        PubSub._name_to_call[name] = call
      else:
        call = PubSub._name_to_call[name]

    with ops.name_scope(f'{name}/{self._rank}'):
      if self._root_rank != self._rank:
        with ops.device(self._bcast_device):
          return collective_ops.broadcast_recv(
            shape, dtype,
            len(self._devices),
            self.channel,
            call)
      value = fn()
      with ops.device(self._bcast_device):
        bcast_send = collective_ops.broadcast_send(
          value, shape, dtype,
          len(self._devices),
          self.channel,
          call)
      with ops.control_dependencies([bcast_send]):
        return array_ops.identity(value)


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
    name = name or Context.get().options.comm_pool_name
    if name not in cls._instances:
      cls._instances[name] = cls()
    return cls._instances[name]

  def __init__(self, impl=None, capacity=None):
    r'''Initialize the communciator pool.

    Args:
      capacity: Capacity of the communciator pool.
    '''
    capacity = capacity or Context.get().options.comm_pool_capacity
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
      local_gpus = [
        device_util.canonicalize(f'/gpu:{d}')
        for d in xrange(Context.get().num_gpus)]
      with ops.device(f'/gpu:{Context.get().rank}'):
        return Communicator.build(
          ops.get_default_graph().get_name_scope(),
          local_gpus,
          impl=self._impl)
    return Communicator.build(
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

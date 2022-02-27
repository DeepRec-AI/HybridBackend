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

import abc

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import CollectiveOps


class Communicator(object):  # pylint: disable=useless-object-inheritance
  r'''A communicator for collective communication.
  '''
  class Resource(object):  # pylint: disable=useless-object-inheritance
    r'''Resource object of a communciator.
    '''
    def __init__(self, comm):
      self._comm = comm

    @property
    def name(self):
      r'''Name of the communicator.
      '''
      return self._comm.name

    @property
    def handle(self):
      r'''Resource handle of the communicator.
      '''
      return self._comm._handle.op  # pylint: disable=protected-access

    @property
    def create(self):
      r'''Resource creation op of the communicator.
      '''
      return self._comm._create_op  # pylint: disable=protected-access

    @property
    def is_initialized(self):
      r'''Resource creation check op of the communicator.
      '''
      return self._comm._is_initialized_op  # pylint: disable=protected-access

  _registry = {}

  @classmethod
  def register(cls, impl):
    cls._registry[impl.NAME] = impl

  @classmethod
  def as_type(cls, impl=None):
    r'''Get communicator implementation type.
    '''
    if impl:
      return impl
    impl_type = Context.get().param(
        'comm_default',
        default='NCCL',
        env='HB_COMM_DEFAULT')
    if impl_type not in cls._registry:
      raise ValueError(
          f'HB_COMM_DEFAULT is invalid: {impl_type}')
    return cls._registry[impl_type]

  @classmethod
  def create(cls, shared_name, devices, impl=None, **kwargs):
    r'''Create a communicator.

    Args:
      shared_name: shared name of the communicator.
      devices: devices of the communicator.
      impl: implementation class for communication.
      kwargs: (Optional.) key-value arguments.
    '''
    if not devices:
      raise ValueError('devices must be provided')
    devices = Context.canonicalize(devices)
    if not devices:
      raise ValueError('Devices must not be empty.')
    return cls.as_type(impl)(shared_name, devices=devices, **kwargs)

  def __init__(self, shared_name, devices, **kwargs):
    r'''Constructs a communicator instance.

    Args:
      shared_name: shared name of the communicator.
      devices: devices of the communicator.
      kwargs: (Optional.) key-value arguments.
    '''
    if shared_name:
      shared_name = shared_name.replace(':', '_').replace('/', '_')
    else:
      shared_name = ops.get_default_graph().unique_name('communicator')
    self._shared_name = shared_name
    self._device = Context.current_device()
    if self._device not in devices:
      raise ValueError(
          f'Current device {self._device} not in devices {devices}')
    self._devices = devices
    self._kwargs = kwargs
    if len(self._devices) > 1:
      ops.add_to_collection(
          ops.GraphKeys.LOCAL_RESOURCES,
          self._build_resource())
    self._default_wire_dtype_for_float = Context.get().param(
        'comm_wire_dtype_for_float',
        default=dtypes.float32,
        env='HB_COMM_WIRE_DTYPE_FOR_FLOAT',
        parser=dtypes.as_dtype)

  @abc.abstractmethod
  def _build_handle(self):
    pass

  @abc.abstractmethod
  def _build_create_op(self):
    pass

  @abc.abstractmethod
  def _build_is_initialized_op(self):
    pass

  def _build_resource(self):
    r'''Create NcclCommunicator.Resource.
    '''
    with ops.name_scope(self.scope):
      with ops.control_dependencies(None):
        self._handle = self._build_handle()
        self._create_op = self._build_create_op()
        self._is_initialized_op = self._build_is_initialized_op()
        return Communicator.Resource(self)

  @property
  def shared_name(self):
    r'''Shared name of the communicator.
    '''
    return self._shared_name

  @property
  def name(self):
    r'''Name of the communicator instance.
    '''
    return f'{self.shared_name}/replicas/{self.rank}'

  @property
  def device(self):
    r'''Device of the communicator instance.
    '''
    return self._device

  @property
  def devices(self):
    r'''All devices of the communicator.
    '''
    return self._devices

  @property
  def size(self):
    r'''Size of the communicator.
    '''
    return len(self.devices)

  @property
  def rank(self):
    r'''Rank of the communicator instance.
    '''
    return self.devices.index(self.device)

  @property
  def scope(self):
    r'''Name scope of the communicator instance.
    '''
    return f'{self.shared_name}/{self.rank}'

  @property
  def handle(self):
    r'''Handle to the communicator instance.
    '''
    return self._handle

  # All-to-one computation

  @abc.abstractmethod
  def _reduce(self, value, reduce_op, root_rank, name):
    pass

  def reduce(self, value, reduce_op=CollectiveOps.SUM, root_rank=0, name=None):
    r'''Reduce values across devices to the root device.

    Args:
      value: Value on current device to be reduced and scattered.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.
      root_rank: Rank of broadcast root.
      name: Name of the op.

    Returns:
      Reduced and scattered value on current device.
    '''
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._reduce(value, reduce_op, root_rank, name)

  # All-to-all computation

  @abc.abstractmethod
  def _reduce_scatter(self, value, reduce_op, name):
    pass

  def reduce_scatter(self, value, reduce_op=CollectiveOps.SUM, name=None):
    r'''Reduce values across devices and scatter the result to all devices.

    Args:
      value: Value on current device to be reduced and scattered.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.
      name: Name of the op.

    Returns:
      Reduced and scattered value on current device.
    '''
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._reduce_scatter(value, reduce_op, name)

  @abc.abstractmethod
  def _allreduce(self, value, reduce_op, name):
    pass

  def allreduce(self, value, reduce_op=CollectiveOps.SUM, name=None):
    r'''Reduce values across devices.

    Args:
      value: Value to be allreduced.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.
      name: Name of the op.

    Returns:
      Allreduced value.
    '''
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._allreduce(value, reduce_op, name)

  # One-to-all data movement

  @abc.abstractmethod
  def _broadcast(self, value, root_rank, name):
    pass

  def broadcast(self, value, root_rank=0, name=None):
    r'''Broadcast value across devices.

    Args:
      value: Value to broadcast.
      root_rank: Rank of broadcast root.
      name: Name of the op.

    Returns:
      Broadcasted value.
    '''
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._broadcast(value, root_rank, name)

  @abc.abstractmethod
  def _scatter(self, value, root_rank, name):
    pass

  def scatter(self, value, root_rank=0, name=None):
    r'''Scatter value on root device to all devices.

    Args:
      value: Value on current device to be scattered, no use for non-root.
      root_rank: Rank of root scatter device.
      name: Name of the op.

    Returns:
      Scattered value on current device.
    '''
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._scatter(value, root_rank, name)

  # All-to-one data movement

  @abc.abstractmethod
  def _gather(self, value, root_rank, name):
    pass

  def gather(self, value, root_rank=0, name=None):
    r'''Gather all values across devices to root device.

    Args:
      value: Value on current device to be gathered.
      root_rank: Rank of root gather device.
      name: Name of the op.

    Returns:
      Gathered value on root, None or controlling object on non-root.
    '''
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._gather(value, root_rank, name)

  @abc.abstractmethod
  def _gatherv(self, value, root_rank, name):
    pass

  def gatherv(self, value, root_rank=0, name=None):
    r'''Gather all values with varying sizes across devices to root device.

    Args:
      value: Value on current device to be gathered.
      root_rank: Rank of root gather device.
      name: Name of the op.

    Returns:
      Gathered value on root, None or controlling object on non-root.
    '''
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._gatherv(value, root_rank, name)

  # All-to-all data movement

  @abc.abstractmethod
  def _allgather(self, value, name):
    pass

  def allgather(self, value, name=None):
    r'''Gather all values across devices to all devices.

    Args:
      value: Value on current device to be gathered.
      varying_size: Supposing all value sizes on devices are not equal.
      name: Name of the op.

    Returns:
      Gathered value.
    '''
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._allgather(value, name)

  @abc.abstractmethod
  def _allgatherv(self, value, name):
    pass

  def allgatherv(self, value, name=None):
    r'''Gather all values across devices to all devices.

    Args:
      value: Value on current device to be gathered.
      varying_size: Supposing all value sizes on devices are not equal.
      name: Name of the op.

    Returns:
      Gathered value.
    '''
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._allgatherv(value, name)

  @abc.abstractmethod
  def _alltoall(self, value, name, **kwargs):
    pass

  def alltoall(self, value, name=None, **kwargs):
    r'''Shuffle value partitions across devices.

    Args:
      value: Value to be sent to other devices.
      name: Name of the op.

    Returns:
      received value partitions from other devices.
    '''
    if self.size == 1:
      return value

    kwargs.setdefault(
        'wire_dtype_for_float', self._default_wire_dtype_for_float)
    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._alltoall(value, name, **kwargs)

  @abc.abstractmethod
  def _alltoallv(self, value, sizes, common_shape, name, **kwargs):
    pass

  def alltoallv(self, value, sizes, common_shape=None, name=None, **kwargs):
    r'''Shuffle value partitions with varying sizes across devices.

    Args:
      value: Value to be sent to other devices.
      common_shape: common shape of tensors in value.
      name: Name of the op.

    Returns:
      received value partitions from other devices.
    '''
    if self.size == 1:
      return value, sizes

    if common_shape is None:
      common_shape = {}

    kwargs.setdefault(
        'wire_dtype_for_float', self._default_wire_dtype_for_float)
    with ops.name_scope(self.scope):
      with ops.device(value.device):
        return self._alltoallv(value, sizes, common_shape, name, **kwargs)

  @abc.abstractmethod
  def _alltoallw(self, values, common_shape, name, **kwargs):
    pass

  def alltoallw(self, values, common_shape=None, name=None, **kwargs):
    r'''Shuffle values with varying sizes and types across devices.

    Args:
      values: List of values to be sent to other devices.
      common_shape: common shape of tensors in value.
      name: Name of the op.

    Returns:
      received values from other devices.
    '''
    if self.size == 1:
      return values

    if common_shape is None:
      common_shape = {}

    if not isinstance(values, (list, tuple)):
      raise ValueError("Only list or tuple allowed.")
    if len(values) != self.size:
      raise ValueError('Number of values must be same to number of devices')
    if len({v.device for v in values}) > 1:
      raise ValueError('Values must be placed at same device')
    values_device = values[0].device

    kwargs.setdefault(
        'wire_dtype_for_float', self._default_wire_dtype_for_float)
    with ops.name_scope(self.scope):
      with ops.device(values_device):
        return self._alltoallw(values, common_shape, name, **kwargs)

  @abc.abstractmethod
  def _group_alltoallw(self, group_values, common_shapes, name, **kwargs):
    pass

  def group_alltoallw(
      self, group_values, common_shapes=None, name=None, **kwargs):
    r'''Grouped alltoallw

    Args:
      values: List of values to be sent to other devices.
      common_shape: common shape of tensors in value.
      name: Name of the op.

    Returns:
      received values from other devices.
    '''
    if self.size == 1:
      return group_values
    if common_shapes is None:
      common_shapes = [{} for _ in group_values]

    if len({v.device for v in sum(group_values, [])}) > 1:
      raise ValueError('all inputs must be placed at same device')
    for values in group_values:
      if len(values) != self.size:
        raise ValueError('Number of inputs must be same to devices')
    if common_shapes is None:
      common_shapes = [{} for _ in group_values]
    if not group_values:
      return group_values

    kwargs.setdefault(
        'wire_dtype_for_float', self._default_wire_dtype_for_float)
    with ops.name_scope(self.scope):
      with ops.device(group_values[0][0].device):
        return self._group_alltoallw(
            group_values, common_shapes, name, **kwargs)

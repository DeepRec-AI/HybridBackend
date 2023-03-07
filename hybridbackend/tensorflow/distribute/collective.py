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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient

from hybridbackend.tensorflow.distribute.ops import CollectiveOps
from hybridbackend.tensorflow.distribute.ops import Topology
from hybridbackend.tensorflow.distribute.rpc import RpcCollective
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.view import OperationLike


class Collective(object):  # pylint: disable=useless-object-inheritance
  r'''Resource for collective communication.
  '''
  def __init__(self, shared_name, handle, create_op, is_initialized_op):
    r'''Constructs an instance for collective communication.

    Args:
      shared_name: shared name for collective communication.
      handle: Handle to the resource.
      create_op: Operation to create the resource.
      is_initialized_op: Operation to check if the reousrce is initialized.
    '''
    self._shared_name = shared_name
    self._handle = handle
    self._create_op = create_op
    self._is_initialized_op = is_initialized_op

  @classmethod
  def _resource_handle(cls, shared_name):
    r'''Resource handle op.
    '''
    return (
      OperationLike('CollectiveHandleOp')
      .returns_tensor({}, dtypes.resource)
      .finalize(shared_name=shared_name))

  @classmethod
  def _resource_get_id(cls, dtype, shape):
    r'''Resource get_id op.
    '''
    return (
      OperationLike('GetCollectiveId').returns_tensor(shape, dtype).finalize())

  @classmethod
  def _resource_create(cls, handle, collective_id, shared_name):
    r'''Resource get_id op.
    '''
    return (
      OperationLike('CreateCollective')
      .finalize(
        handle, collective_id,
        world_size=Context.get().world_size,
        local_size=Context.get().local_world_size,
        rank=Context.get().rank,
        shared_name=shared_name))

  @classmethod
  def _resource_is_initialized(cls, handle):
    r'''Resource is_initialized op.
    '''
    return (
      OperationLike('IsCollectiveInitialized')
      .returns_tensor({}, dtypes.bool)
      .finalize(handle))

  @classmethod
  def get(cls):
    r'''Get or create an instance.
    '''
    ctx = Context.get()
    if ctx.world_size == 1:
      return cls('', None, None, None)

    collective_resources = ops.get_collection_ref(cls.__name__)
    if collective_resources:
      return collective_resources[0]

    shared_name = ops.get_default_graph().unique_name('collective')
    with ops.name_scope(f'{shared_name}/replicas/{ctx.rank}/'):
      with ops.control_dependencies(None):
        handle = cls._resource_handle(shared_name)
        collective_id_dtype = dtypes.int64
        collective_id_shape = tensor_shape.TensorShape([16])  # 128 / 8
        collective_id = (
          RpcCollective(ctx.devices, ctx.rank).broadcast(
            lambda: cls._resource_get_id(
              collective_id_dtype, collective_id_shape),
            collective_id_dtype, collective_id_shape,
            name=f'{shared_name}_id/rpc_broadcast'))
        create_op = cls._resource_create(handle, collective_id, shared_name)
        is_initialized_op = cls._resource_is_initialized(handle)
        coll = cls(shared_name, handle, create_op, is_initialized_op)
        ops.add_to_collection(ops.GraphKeys.LOCAL_RESOURCES, coll)
        collective_resources.append(coll)
        return coll

  @property
  def shared_name(self):
    r'''Shared name of the communicator.
    '''
    return self._shared_name

  @property
  def name(self):
    r'''Name of the communicator instance.
    '''
    return f'{self.shared_name}/replicas/{Context.get().rank}/'

  @property
  def handle(self):
    r'''Handle op to the communicator instance.
    '''
    return self._handle.op

  @property
  def create(self):
    r'''Resource creation op of the communicator.
    '''
    return self._create_op  # pylint: disable=protected-access

  @property
  def is_initialized(self):
    r'''Resource creation check op of the communicator.
    '''
    return self._is_initialized_op  # pylint: disable=protected-access

  # All to all operations

  def _allreduce(self, value, reduce_op, name):
    r'''Reduce values across devices.

    Args:
      value: Value to be allreduced.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.
      name: Name of the op.

    Returns:
      Allreduced value.
    '''
    if Context.get().world_size == 1:
      return value

    with ops.device(value.device):
      with ops.name_scope(self.name):
        return (
          OperationLike('CollectiveAllreduce')
          .returns_tensor(value.shape, value.dtype)
          .finalize(self._handle, value, reduce_op=reduce_op, name=name))

  def allreduce(self, value, reduce_op=None, name=None):
    r'''Reduce values across devices.

    Args:
      value: Value to be allreduced.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.
      name: Name of the op.

    Returns:
      Allreduced value.
    '''
    if reduce_op is None:
      reduce_op = CollectiveOps.SUM
    if name is None:
      name = 'allreduce'

    @custom_gradient.custom_gradient
    def allreduce_fn(value):
      reduced = self._allreduce(
        value, reduce_op=reduce_op, name=name)

      def grad_fn(value_grad):
        r'''Gradient function for fn.
        '''
        if reduce_op != CollectiveOps.SUM:
          raise NotImplementedError(
            'Only reduce_op=SUM is supported for gradients computation.')
        with ops.control_dependencies([reduced]):
          grad_out = self._allreduce(
            value_grad, reduce_op=reduce_op, name=f'gradients/{name}')
        return grad_out
      return reduced, grad_fn

    return allreduce_fn(value)

  def _alltoall(self, value, topology, wire_dtype, name):
    r'''Shuffle value partitions across devices.

    Args:
      value: Value to be sent to other devices.
      name: Name of the op.
      topology: Communication topology.
      wire_dtype: Data type for communication if possible.

    Returns:
      received value partitions from other devices.
    '''
    if Context.get().world_size == 1:
      return value

    with ops.device(value.device):
      with ops.name_scope(self.name):
        return (
          OperationLike('CollectiveAlltoall')
          .returns_tensor(value.shape, value.dtype)
          .finalize(
            self._handle, value,
            wire_dtype=wire_dtype,
            topology=topology,
            name=name))

  def _alltoallv(
      self, value, sizes, common_shape,
      topology, wire_dtype, name):
    r'''Shuffle value partitions across devices.

    Args:
      value: Value to be sent to other devices.
      sizes: Sizes of parts in the value.
      common_shape: Common shape of values.
      topology: Communication topology.
      wire_dtype: Data type for communication if possible.

    Returns:
      received value partitions from other devices.
    '''
    if Context.get().world_size == 1:
      return value

    with ops.device(value.device):
      with ops.name_scope(self.name):
        return (
          OperationLike('CollectiveAlltoallv')
          .returns_tensors(
            tensor_spec.TensorSpec(
              tensor_shape.TensorShape([None]).concatenate(common_shape),
              value.dtype),
            tensor_spec.TensorSpec(sizes.shape, sizes.dtype))
          .finalize(
            self._handle, value, sizes,
            wire_dtype=wire_dtype,
            common_shape=tensor_shape.TensorShape(common_shape),
            topology=topology,
            name=name))

  def alltoall(
      self, value,
      sizes=None,
      common_shape=None,
      topology=Topology.ALL,
      name=None):
    r'''Shuffle value partitions across devices.

    Args:
      value: Value to be sent to other devices.
      sizes: Sizes of splits in corresponding value.
      common_shape: common shape of tensors in value.
      topology: Communication topology.
      name: Name of the op.

    Returns:
      received value partitions from other devices.
    '''
    if common_shape is None:
      common_shape = {}
    wire_dtype = Context.get().options.comm_wire_dtype
    if wire_dtype is None:
      wire_dtype = dtypes.float32
    gradient_wire_dtype = Context.get().options.comm_gradient_wire_dtype
    if gradient_wire_dtype is None:
      gradient_wire_dtype = wire_dtype
    if name is None:
      name = 'alltoall'

    @custom_gradient.custom_gradient
    def alltoall_fn(value):
      exchanged = self._alltoall(
        value,
        topology=topology,
        wire_dtype=wire_dtype,
        name=name)

      def grad_fn(value_grad):
        r'''Gradient function for fn.
        '''
        value_grad = ops.convert_to_tensor(
          value_grad, name=f'gradients/{name}/to_dense')
        with ops.control_dependencies([exchanged]):
          grad_out = self._alltoall(
            value_grad,
            topology=topology,
            wire_dtype=gradient_wire_dtype,
            name=f'gradients/{name}')
        return grad_out
      return exchanged, grad_fn

    if sizes is None:
      return alltoall_fn(value)

    @custom_gradient.custom_gradient
    def alltoallv_fn(value, sizes):
      exchanged_value, exchanged_sizes = self._alltoallv(
        value, sizes,
        common_shape=tensor_shape.TensorShape(common_shape),
        topology=topology,
        wire_dtype=wire_dtype,
        name=name)

      def grad_fn(value_grad, sizes_grad):
        r'''Gradient function for fn.
        '''
        del sizes_grad
        value_grad = ops.convert_to_tensor(
          value_grad, name=f'gradients/{name}/to_dense')
        with ops.control_dependencies([exchanged_value, exchanged_sizes]):
          value_grad_out, sizes_grad_out = self._alltoallv(
            value_grad, exchanged_sizes,
            common_shape=tensor_shape.TensorShape(common_shape),
            topology=topology,
            wire_dtype=gradient_wire_dtype,
            name=f'gradients/{name}')
        return value_grad_out, sizes_grad_out
      return (exchanged_value, exchanged_sizes), grad_fn

    return alltoallv_fn(value, sizes)

  # One to all / all to all operations

  def _broadcast(self, value, root_rank, name):
    r'''Broadcast value across devices.

    Args:
      value: Value to broadcast.
      root_rank: Rank of broadcast root.
      name: Name of the op.

    Returns:
      Broadcasted value.
    '''
    if Context.get().world_size == 1:
      return value

    with ops.device(value.device):
      with ops.name_scope(self.name):
        return (
          OperationLike('CollectiveBroadcast')
          .returns_tensor(value.shape, value.dtype)
          .finalize(self._handle, value, root_rank=root_rank, name=name))

  def broadcast(self, value, root_rank=None, name=None):
    r'''Broadcast value across devices.

    Args:
      value: Value to broadcast.
      root_rank: Rank of broadcast root.
      name: Name of the op.

    Returns:
      Broadcasted value.
    '''
    if root_rank is None:
      root_rank = 0
    if name is None:
      name = 'broadcast'

    return array_ops.stop_gradient(
      self._broadcast(value, root_rank=root_rank, name=name))

  def _allgather(self, value, name):
    r'''Gather all values across devices to all devices.

    Args:
      value: Value on current device to be gathered.
      varying_size: Supposing all value sizes on devices are not equal.
      name: Name of the op.

    Returns:
      Gathered value.
    '''
    if Context.get().world_size == 1:
      return value

    with ops.device(value.device):
      with ops.name_scope(self.name):
        input_shape = value.shape
        if input_shape.rank < 1:
          input_shape = tensor_shape.TensorShape([1]).concatenate(input_shape)
        output_shape = tensor_shape.TensorShape([
          Context.get().world_size * input_shape.dims[0],
          *input_shape.dims[1:]])
        return (
          OperationLike('CollectiveAllgather')
          .returns_tensor(output_shape, value.dtype)
          .finalize(self._handle, value, name=name))

  def _allgatherv(self, value, name):
    r'''Gather all values across devices to all devices.

    Args:
      value: Value on current device to be gathered.
      varying_size: Supposing all value sizes on devices are not equal.
      name: Name of the op.

    Returns:
      Gathered value.
    '''
    if Context.get().world_size == 1:
      return value

    with ops.device(value.device):
      with ops.name_scope(self.name):
        input_shape = value.shape
        if input_shape.rank < 1:
          input_shape = tensor_shape.TensorShape([1]).concatenate(input_shape)
        output_shape = tensor_shape.TensorShape([None, *input_shape.dims[1:]])
        return (
          OperationLike('CollectiveAllgatherv')
          .returns_tensor(output_shape, value.dtype)
          .finalize(self._handle, value, name=name))

  def allgather(self, value, varying_size=True, name=None):
    r'''Gather all values across devices to all devices.

    Args:
      value: Value on current device to be gathered.
      varying_size: Supposing all value sizes on devices are not equal.
      name: Name of the op.

    Returns:
      Gathered value.
    '''
    if name is None:
      name = 'allgather'

    if varying_size:
      return array_ops.stop_gradient(
        self._allgatherv(value, name=name))

    return array_ops.stop_gradient(
      self._allgather(value, name=name))


def active_size(topology):
  r'''Obtain the size of active ranks from a topology.

  Args:
    topology: Communication topology.

  Returns:
    size of active ranks.
  '''
  if topology == Topology.INTRA_NODE:
    return Context.get().local_world_size
  if topology == Topology.INTER_NODE:
    return int(
      Context.get().world_size // Context.get().local_world_size)
  return Context.get().world_size


def allreduce(value, reduce_op=None, name=None):
  r'''Reduce values across devices.

  Args:
    value: Value to be allreduced.
    reduce_op: Reduction ops: SUM, PROD, MAX or MIN.
    name: Name of the op.

  Returns:
    Allreduced value.
  '''
  return Collective.get().allreduce(
    value,
    reduce_op=reduce_op,
    name=name)


def alltoall(
    value, sizes=None, common_shape=None, topology=Topology.ALL, name=None):
  r'''Shuffle value partitions across devices.

  Args:
    value: Value to be sent to other devices.
    sizes: Sizes of splits in corresponding value.
    common_shape: common shape of tensors in value.
    topology: Communication topology.
    name: Name of the op.

  Returns:
    received value partitions from other devices.
  '''
  return Collective.get().alltoall(
    value,
    sizes=sizes,
    common_shape=tensor_shape.TensorShape(common_shape),
    topology=topology,
    name=name)


def broadcast(value, root_rank=None, name=None):
  r'''Broadcast value across devices.

  Args:
    value: Value to broadcast.
    root_rank: Rank of broadcast root.
    name: Name of the op.

  Returns:
    Broadcasted value.
  '''
  return Collective.get().broadcast(value, root_rank=root_rank, name=name)


def allgather(value, varying_size=True, name=None):
  r'''Gather all values across devices to all devices.

  Args:
    value: Value on current device to be gathered.
    varying_size: Supposing all value sizes on devices are not equal.
    name: Name of the op.

  Returns:
    Gathered value.
  '''
  return Collective.get().allgather(value, varying_size=varying_size, name=name)

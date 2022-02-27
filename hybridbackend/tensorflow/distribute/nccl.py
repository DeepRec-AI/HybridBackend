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

r'''NCCL based collective commmunication.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging

from hybridbackend.tensorflow.distribute.communicator import Communicator
from hybridbackend.tensorflow.distribute.pubsub import PubSub
from hybridbackend.tensorflow.framework.ops import CollectiveOps
from hybridbackend.tensorflow.pywrap import _ops


ops.NotDifferentiable('GetNcclId')

ops.NotDifferentiable('NcclComm')
ops.NotDifferentiable('CreateNcclComm')
ops.NotDifferentiable('IsNcclCommInitialized')


@ops.RegisterGradient('ReduceWithNcclComm')
def _reduce_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL reduce op.
  '''
  comm = op.inputs[0]
  grad_in = args[0]
  reduce_op = op.get_attr("reduce_op")
  root_rank = op.get_attr("root_rank")
  if reduce_op != CollectiveOps.SUM:
    raise NotImplementedError(
        'Only reduce_op=SUM is supported for gradients computation.')
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.broadcast_with_nccl_comm(
          comm, grad_in, root_rank=root_rank)
  return None, grad_out


@ops.RegisterGradient('ReduceScatterWithNcclComm')
def _reduce_scatter_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL reduce scatter op.
  '''
  comm = op.inputs[0]
  grad_in = args[0]
  reduce_op = op.get_attr("reduce_op")
  if reduce_op != CollectiveOps.SUM:
    raise NotImplementedError(
        'Only reduce_op=SUM is supported for gradients computation.')
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.allgather_with_nccl_comm(comm, grad_in)
  return None, grad_out


@ops.RegisterGradient('AllreduceWithNcclComm')
def _allreduce_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL allreduce op.
  '''
  comm = op.inputs[0]
  grad_in = args[0]
  reduce_op = op.get_attr("reduce_op")
  if reduce_op != CollectiveOps.SUM:
    raise NotImplementedError(
        'Only reduce_op=SUM is supported for gradients computation.')
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.allreduce_with_nccl_comm(
          comm, grad_in, reduce_op=reduce_op)
  return None, grad_out


ops.NotDifferentiable('BroadcastWithNcclComm')
ops.NotDifferentiable('ScatterWithNcclComm')
ops.NotDifferentiable('GatherWithNcclComm')
ops.NotDifferentiable('GathervWithNcclComm')


@ops.RegisterGradient('AllgatherWithNcclComm')
def _allgather_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL allgather op.
  '''
  comm = op.inputs[0]
  grad_in = args[0]
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.reduce_scatter_with_nccl_comm(
          comm, grad_in, reduce_op=CollectiveOps.SUM)
  return None, grad_out


ops.NotDifferentiable('AllgathervWithNcclComm')


@ops.RegisterGradient('AlltoallWithNcclComm')
def _alltoall_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL alltoall op.
  '''
  comm = op.inputs[0]
  grad_in = args[0]
  wire_dtype_for_float = op.get_attr("wire_dtype_for_float")
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.alltoall_with_nccl_comm(
          comm, grad_in,
          wire_dtype_for_float=wire_dtype_for_float)
  return None, grad_out


@ops.RegisterGradient('AlltoallvWithNcclComm')
def _nccl_alltoallv_grad(op, *args):
  r'''Gradient for NCCL alltoallv op.
  '''
  comm = op.inputs[0]
  grad_in = list(args)[0]
  grad_sizes_in = op.outputs[1]
  common_shape = op.get_attr("common_shape")
  wire_dtype_for_float = op.get_attr("wire_dtype_for_float")
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out, grad_sizes_out = _ops.alltoallv_with_nccl_comm(
          comm, grad_in, grad_sizes_in,
          wire_dtype_for_float=wire_dtype_for_float,
          common_shape=common_shape)
  return None, grad_out, grad_sizes_out


@ops.RegisterGradient('AlltoallwWithNcclComm')
def _alltoallw_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL alltoallw op.
  '''
  comm = op.inputs[0]
  common_shape = op.get_attr("common_shape")
  wire_dtype_for_float = op.get_attr("wire_dtype_for_float")
  grad_in = list(args)
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.alltoallw_with_nccl_comm(
          comm, grad_in,
          wire_dtype_for_float=wire_dtype_for_float,
          common_shape=common_shape)
  return [None] + grad_out

@ops.RegisterGradient('GroupAlltoallwWithNcclComm')
def _group_alltoallw_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL group_alltoallw op.
  '''
  comm_handle = op.inputs[0]
  group_size = op.get_attr("group_size")
  rank = op.get_attr("rank")
  wire_dtype_for_float = op.get_attr("wire_dtype_for_float")
  common_shapes = op.get_attr("common_shapes")
  grad_in = list(args)
  with ops.device(op.device):
    with ops.control_dependencies([op]):
      grad_out = _ops.group_alltoallw_with_nccl_comm(
          comm_handle, grad_in,
          group_size=group_size,
          rank=rank,
          wire_dtype_for_float=wire_dtype_for_float,
          common_shapes=common_shapes)
  return [None] + grad_out

class NcclCommunicator(Communicator):
  r'''A communicator using NCCL.
  '''
  NAME = 'NCCL'

  @classmethod
  def compute_pool_capacity(cls, capacity=None):
    if capacity is not None and capacity != 1:
      logging.warning('Multiple-communicators pooling is unsafe.')
    return capacity or 1

  def _build_handle(self):
    return _ops.nccl_comm_handle_op(shared_name=self.shared_name)

  def _build_create_op(self):
    nccl_id = PubSub(self.devices)(
        _ops.get_nccl_id,
        tensor_shape.TensorShape([16]), # 128 / 8
        dtypes.int64,
        name=f'{self.shared_name}/nccl_id')
    return _ops.create_nccl_comm(
        self.handle, nccl_id,
        size=self.size,
        rank=self.rank,
        shared_name=self.shared_name)

  def _build_is_initialized_op(self):
    return _ops.is_nccl_comm_initialized(self.handle)

  def _reduce(self, value, reduce_op, root_rank, name):
    return _ops.reduce_with_nccl_comm(
        self.handle, value,
        reduce_op=reduce_op,
        root_rank=root_rank,
        name=name)

  def _reduce_scatter(self, value, reduce_op, name):
    return _ops.reduce_scatter_with_nccl_comm(
        self.handle, value,
        reduce_op=reduce_op,
        name=name)

  def _allreduce(self, value, reduce_op, name):
    return _ops.allreduce_with_nccl_comm(
        self.handle, value,
        reduce_op=reduce_op,
        name=name)

  def _broadcast(self, value, root_rank, name):
    return _ops.broadcast_with_nccl_comm(
        self.handle, value,
        root_rank=root_rank,
        name=name)

  def _scatter(self, value, root_rank, name):
    raise NotImplementedError

  def _gather(self, value, root_rank, name):
    raise NotImplementedError

  def _gatherv(self, value, root_rank, name):
    raise NotImplementedError

  def _allgather(self, value, name):
    return _ops.allgather_with_nccl_comm(self.handle, value, name=name)

  def _allgatherv(self, value, name):
    return _ops.allgatherv_with_nccl_comm(self.handle, value, name=name)

  def _alltoall(self, value, name, **kwargs):
    return _ops.alltoall_with_nccl_comm(
        self.handle, value,
        wire_dtype_for_float=kwargs.pop('wire_dtype_for_float', dtypes.float32),
        name=name)

  def _alltoallv(self, value, sizes, common_shape, name, **kwargs):
    return _ops.alltoallv_with_nccl_comm(
        self.handle, value, sizes,
        common_shape=common_shape,
        wire_dtype_for_float=kwargs.pop('wire_dtype_for_float', dtypes.float32),
        name=name)

  def _alltoallw(self, values, common_shape, name, **kwargs):
    return _ops.alltoallw_with_nccl_comm(
        self.handle, values,
        common_shape=common_shape,
        wire_dtype_for_float=kwargs.pop('wire_dtype_for_float', dtypes.float32),
        name=name)

  def _group_alltoallw(self, group_values, common_shapes, name, **kwargs):
    flatten_results = _ops.group_alltoallw_with_nccl_comm(
        self._handle,
        sum(group_values, []),
        group_size=len(group_values),
        rank=self.rank,
        wire_dtype_for_float=kwargs.pop('wire_dtype_for_float', dtypes.float32),
        common_shapes=common_shapes)
    results = []
    for v in flatten_results:
      if results and len(results[-1]) < self.size:
        results[-1].append(v)
      else:
        results.append([v])
    return results

Communicator.register(NcclCommunicator)

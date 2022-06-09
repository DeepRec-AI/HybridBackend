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

from hybridbackend.tensorflow.common import oplib as _ops
from hybridbackend.tensorflow.distribute.communicator import CollectiveOps
from hybridbackend.tensorflow.distribute.communicator import Communicator
from hybridbackend.tensorflow.distribute.communicator_lib import PubSub

ops.NotDifferentiable('HbGetNcclId')

ops.NotDifferentiable('HbNcclComm')
ops.NotDifferentiable('HbCreateNcclComm')
ops.NotDifferentiable('HbIsNcclCommInitialized')


@ops.RegisterGradient('HbNcclReduce')
def _reduce_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL reduce op.
  '''
  comm = op.inputs[0]
  grad_in = args[0]
  reduce_op = op.get_attr('reduce_op')
  root_rank = op.get_attr('root_rank')
  if reduce_op != CollectiveOps.SUM:
    raise NotImplementedError(
      'Only reduce_op=SUM is supported for gradients computation.')
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.hb_nccl_broadcast(
        comm, grad_in, root_rank=root_rank)
  return None, grad_out


@ops.RegisterGradient('HbNcclReduceScatter')
def _reduce_scatter_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL reduce scatter op.
  '''
  comm = op.inputs[0]
  grad_in = args[0]
  reduce_op = op.get_attr('reduce_op')
  if reduce_op != CollectiveOps.SUM:
    raise NotImplementedError(
      'Only reduce_op=SUM is supported for gradients computation.')
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.hb_nccl_allgather(comm, grad_in)
  return None, grad_out


@ops.RegisterGradient('HbNcclAllreduce')
def _allreduce_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL allreduce op.
  '''
  comm = op.inputs[0]
  grad_in = args[0]
  reduce_op = op.get_attr('reduce_op')
  if reduce_op != CollectiveOps.SUM:
    raise NotImplementedError(
      'Only reduce_op=SUM is supported for gradients computation.')
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.hb_nccl_allreduce(
        comm, grad_in, reduce_op=reduce_op)
  return None, grad_out


ops.NotDifferentiable('HbNcclBroadcast')
ops.NotDifferentiable('ScatterWithNcclComm')
ops.NotDifferentiable('GatherWithNcclComm')
ops.NotDifferentiable('GathervWithNcclComm')


@ops.RegisterGradient('HbNcclAllgather')
def _allgather_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL allgather op.
  '''
  comm = op.inputs[0]
  grad_in = args[0]
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.hb_nccl_reduce_scatter(
        comm, grad_in, reduce_op=CollectiveOps.SUM)
  return None, grad_out


ops.NotDifferentiable('HbNcclAllgatherv')


@ops.RegisterGradient('HbNcclAlltoall')
def _alltoall_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL alltoall op.
  '''
  comm = op.inputs[0]
  grad_in = args[0]
  wire_dtype = op.get_attr('wire_dtype')
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.hb_nccl_alltoall(
        comm, grad_in,
        wire_dtype=wire_dtype)
  return None, grad_out


@ops.RegisterGradient('HbNcclAlltoallv')
def _nccl_alltoallv_grad(op, *args):
  r'''Gradient for NCCL alltoallv op.
  '''
  comm = op.inputs[0]
  grad_in = list(args)[0]
  grad_sizes_in = op.outputs[1]
  common_shape = op.get_attr('common_shape')
  wire_dtype = op.get_attr('wire_dtype')
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out, grad_sizes_out = _ops.hb_nccl_alltoallv(
        comm, grad_in, grad_sizes_in,
        wire_dtype=wire_dtype,
        common_shape=common_shape)
  return None, grad_out, grad_sizes_out


@ops.RegisterGradient('HbNcclAlltoallvN')
def _alltoallv_n_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL alltoallv_n op.
  '''
  comm = op.inputs[0]
  num_columns = op.get_attr('num_columns')
  wire_dtype = op.get_attr('wire_dtype')
  common_shapes = op.get_attr('common_shapes')
  grad_in = args[:num_columns]
  grad_sizes_in = op.outputs[num_columns:]
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out, _ = _ops.hb_nccl_alltoallv_n(
        comm, grad_in, grad_sizes_in,
        wire_dtype=wire_dtype,
        common_shapes=common_shapes)
  return (None, *grad_out, *[None for _ in range(num_columns)])


@ops.RegisterGradient('HbNcclAlltoallw')
def _alltoallw_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL alltoallw op.
  '''
  comm = op.inputs[0]
  common_shape = op.get_attr('common_shape')
  wire_dtype = op.get_attr('wire_dtype')
  grad_in = list(args)
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.hb_nccl_alltoallw(
        comm, grad_in,
        wire_dtype=wire_dtype,
        common_shape=common_shape)
  return [None] + grad_out


@ops.RegisterGradient('HbNcclAlltoallwN')
def _alltoallw_n_with_nccl_comm_grad(op, *args):
  r'''Gradient for NCCL alltoallw_n op.
  '''
  comm = op.inputs[0]
  num_columns = op.get_attr('num_columns')
  wire_dtype = op.get_attr('wire_dtype')
  common_shapes = op.get_attr('common_shapes')
  grad_in = list(args)
  with ops.device(op.device):
    with ops.control_dependencies(op.outputs):
      grad_out = _ops.hb_nccl_alltoallw_n(
        comm, grad_in,
        num_columns=num_columns,
        wire_dtype=wire_dtype,
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
    return _ops.hb_nccl_comm_handle_op(shared_name=self.shared_name)

  def _build_create_op(self):
    nccl_id = PubSub(self.devices)(
      _ops.hb_get_nccl_id,
      tensor_shape.TensorShape([16]),  # 128 / 8
      dtypes.int64,
      name=f'{self.shared_name}/nccl_id')
    return _ops.hb_create_nccl_comm(
      self.handle, nccl_id,
      size=self.size,
      rank=self.rank,
      shared_name=self.shared_name)

  def _build_is_initialized_op(self):
    return _ops.hb_is_nccl_comm_initialized(self.handle)

  def _reduce(self, value, reduce_op, root_rank, name):
    return _ops.hb_nccl_reduce(
      self.handle, value,
      reduce_op=reduce_op,
      root_rank=root_rank,
      name=name)

  def _reduce_scatter(self, value, reduce_op, name):
    return _ops.hb_nccl_reduce_scatter(
      self.handle, value,
      reduce_op=reduce_op,
      name=name)

  def _allreduce(self, value, reduce_op, name):
    return _ops.hb_nccl_allreduce(
      self.handle, value,
      reduce_op=reduce_op,
      name=name)

  def _broadcast(self, value, root_rank, name):
    return _ops.hb_nccl_broadcast(
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
    return _ops.hb_nccl_allgather(self.handle, value, name=name)

  def _allgatherv(self, value, name):
    return _ops.hb_nccl_allgatherv(self.handle, value, name=name)

  def _alltoall(self, value, wire_dtype, name, **kwargs):
    return _ops.hb_nccl_alltoall(
      self.handle, value,
      wire_dtype=wire_dtype,
      name=name)

  def _alltoallv(self, value, sizes, wire_dtype, common_shape, name, **kwargs):
    return _ops.hb_nccl_alltoallv(
      self.handle, value, sizes,
      common_shape=common_shape,
      wire_dtype=wire_dtype,
      name=name)

  def _alltoallv_n(
      self, values, sizes, wire_dtype, common_shapes, name, **kwargs):
    return _ops.hb_nccl_alltoallv_n(
      self.handle, values, sizes,
      common_shapes=common_shapes,
      wire_dtype=wire_dtype,
      name=name)

  def _alltoallw(self, values, wire_dtype, common_shape, name, **kwargs):
    return _ops.hb_nccl_alltoallw(
      self.handle, values,
      common_shape=common_shape,
      wire_dtype=wire_dtype,
      name=name)

  def _alltoallw_n(
      self, values_list, wire_dtype, common_shapes, name, **kwargs):
    flatten_results = _ops.hb_nccl_alltoallw_n(
      self._handle,
      sum(values_list, []),
      num_columns=len(values_list),
      wire_dtype=wire_dtype,
      common_shapes=common_shapes)
    results = []
    for v in flatten_results:
      if results and len(results[-1]) < self.size:
        results[-1].append(v)
      else:
        results.append([v])
    return results


Communicator.register(NcclCommunicator)

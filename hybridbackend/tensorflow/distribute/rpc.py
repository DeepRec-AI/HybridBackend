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

r'''Classes for RPC based collective communication.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import threading

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops


class RpcCollective(object):  # pylint: disable=useless-object-inheritance
  r'''RPC based collective ops.
  '''
  _lock = threading.Lock()

  _prev_group_key = 1
  _group_key_dict = {}

  _prev_instance_key = 1000000
  _instance_key_dict = {}

  def __init__(self, group_world, group_rank):
    r'''Constructs an RPC based collective instance.

    Args:
      group_world: Involved devices.
      group_rank: Rank of this peer.
    '''
    self._group_devices = sorted(group_world)
    group_name = ','.join(self._group_devices)
    with RpcCollective._lock:
      if group_name not in RpcCollective._group_key_dict:
        group_key = RpcCollective._prev_group_key
        RpcCollective._prev_group_key += 1
        RpcCollective._group_key_dict[group_name] = group_key
    self._group_key = RpcCollective._group_key_dict[group_name]
    self._group_rank = group_rank

  @property
  def group_key(self):
    r'''Key for involved devices.
    '''
    return self._group_key

  @property
  def group_size(self):
    r'''Size of involved devices.
    '''
    return len(self._group_devices)

  @contextlib.contextmanager
  def instance_scope(self, shared_name, name=None):
    r'''Context for an RPC based collective op instance.
    '''
    if name is None:
      name = ops.get_default_graph().unique_name(f'rpc_{shared_name}')
    name = name.replace(':', '_').replace('/', '_')
    with RpcCollective._lock:
      if name not in RpcCollective._instance_key_dict:
        instance_key = RpcCollective._prev_instance_key
        RpcCollective._prev_instance_key += 1
        RpcCollective._instance_key_dict[name] = instance_key
      else:
        instance_key = RpcCollective._instance_key_dict[name]
    with ops.name_scope(f'{name}/replicas/{self._group_rank}/'):
      yield instance_key

  def broadcast(self, fn, dtype, shape, root_rank=None, name=None):
    r'''Broadcast tensor across devices.

    Args:
      fn: Function to generate tensor.
      dtype: Data type of the generated tensor.
      shape: Shape of the generate tensor.
      root_rank: Rank of the root peer.
      name: Name of the op.

    Returns:
      Broadcasted value.
    '''
    if self.group_size == 1:
      return fn()

    if root_rank is None:
      root_rank = 0
    with self.instance_scope('broadcast', name=name) as instance_key:
      if root_rank != self._group_rank:
        with ops.device('/cpu:0'):
          return collective_ops.broadcast_recv(
            shape, dtype,
            self.group_size,
            self.group_key,
            instance_key,
            communication_hint='ring')
      value = fn()
      with ops.device('/cpu:0'):
        bcast_send = collective_ops.broadcast_send(
          value, shape, dtype,
          self.group_size,
          self.group_key,
          instance_key,
          communication_hint='ring')
      with ops.control_dependencies([bcast_send]):
        return array_ops.identity(value)

  def allreduce(self, value, merge_op='Add', final_op='Id', name=None):
    r'''Allreduce tensor across devices.

    Args:
      value: tensor to allreduce.
      merge_op: 'Mul', 'Add'.
      final_op: 'Id', 'Div'.
      name: Name of the op.

    Returns:
      Allreduced value.
    '''
    if self.group_size == 1:
      return value

    with self.instance_scope('allreduce', name=name) as instance_key:
      with ops.device('/cpu:0'):
        return collective_ops.all_reduce(
          value, self.group_size, self.group_key, instance_key,
          merge_op=merge_op,
          final_op=final_op,
          communication_hint='ring')

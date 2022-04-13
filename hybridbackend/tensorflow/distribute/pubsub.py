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

r'''Communication using RPC.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops

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
    self._bcast_device = Context.get().param('pubsub_device', '/cpu:0')
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

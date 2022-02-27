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

r'''Utilities for servers using hybrid parallelsim.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import device as pydev

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.device import device_function


def build_session_config(prototype, *args, **kwargs):
  r'''Creates ConfigProto from prototype.
  '''
  ctx = Context.get()
  if not prototype:
    if 'allow_soft_placement' not in kwargs:
      kwargs['allow_soft_placement'] = True
    prototype = config_pb2.ConfigProto(*args, **kwargs)
  prototype.gpu_options.allow_growth = True
  prototype.gpu_options.force_gpu_compatible = True
  chief = pydev.DeviceSpec.from_string(ctx.devices[0])
  del prototype.device_filters[:]
  prototype.device_filters.append(f'/job:{chief.job}/task:{chief.task}')
  prototype.device_filters.append(
      f'/job:{ctx.task_type}/task:{ctx.task_id}')
  prototype.experimental.collective_group_leader = \
      f'/job:{chief.job}/replica:{chief.replica}/task:{chief.task}'
  return prototype


def device_setter():
  r'''Return a device string to use when building a Graph for replicas.

  Returns:
    device_str: device placement.
  '''
  return device_function

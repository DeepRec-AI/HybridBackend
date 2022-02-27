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

r'''Utilities for device placement.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import device as pydev

from hybridbackend.tensorflow.framework.context import Context


def device_function(op):
  r'''Device function for HybridBackend.

  Args:
    op: Operator to place.

  Returns:
    device_string: device placement.
  '''
  ctx = Context.get()
  current_device = pydev.DeviceSpec.from_string(op.device or '')
  if ctx.has_gpu:
    local_device = '/gpu:0'
  else:
    local_device = '/cpu:0'
  worker_device = pydev.DeviceSpec.from_string(
    f'/job:{ctx.task_type}/task:{ctx.task_id}{local_device}')
  if hasattr(worker_device, 'merge_from'):
    worker_device.merge_from(current_device)
  else:
    worker_device = worker_device.make_merged_spec(current_device)
  return worker_device.to_string()

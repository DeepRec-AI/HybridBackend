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

r'''ConfigProto related functions.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute import multi_worker_util

from hybridbackend.tensorflow.framework.context import Context


def wraps_session_config(session_config, *args, **kwargs):
  r'''Wraps ConfigProto for distributed training.
  '''
  if not session_config:
    kwargs.setdefault('allow_soft_placement', True)
    session_config = config_pb2.ConfigProto(*args, **kwargs)
  session_config.gpu_options.allow_growth = True
  session_config.gpu_options.force_gpu_compatible = True
  if not session_config.device_filters:
    cluster_spec = Context.get().cluster_spec
    task_type = Context.get().task_type
    task_id = Context.get().task_id
    if cluster_spec is None:
      session_config.isolate_session_state = True
      return session_config
    session_config.isolate_session_state = False
    del session_config.device_filters[:]
    if task_type in ('chief', 'worker'):
      session_config.device_filters.extend([
        '/job:ps', '/job:chief', f'/job:{task_type}/task:{task_id}'])
      session_config.experimental.collective_group_leader = (
        multi_worker_util.collective_leader(cluster_spec, task_type, task_id))
    elif task_type == 'evaluator':
      session_config.device_filters.append(f'/job:{task_type}/task:{task_id}')
  return session_config


def get_session_config(*args, **kwargs):
  r'''Creates ConfigProto for distributed training.
  '''
  return wraps_session_config(None, *args, **kwargs)

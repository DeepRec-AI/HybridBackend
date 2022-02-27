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

r'''Servers using hybrid parallelism.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.training.server_lib import build_session_config
from hybridbackend.tensorflow.training.server_lib import device_setter


def wraps_server(cls):
  r'''Decorator to create hybridbackend server class.
  '''
  class HybridBackendServer(cls):
    r'''An in-process TensorFlow server, for use in distributed training.
    '''
    def __init__(self, server_or_cluster_def=None, **kwargs):
      r'''Creates a new server with the given definition.
      '''
      if server_or_cluster_def is None:
        server_or_cluster_def = Context.get().cluster_spec
      if server_or_cluster_def is None:
        self._is_local = True
        return
      self._is_local = False
      kwargs['job_name'] = Context.get().task_type
      kwargs['task_index'] = Context.get().task_id
      kwargs['config'] = build_session_config(kwargs.pop('config', None))
      super().__init__(server_or_cluster_def, **kwargs)

    @property
    def target(self):
      r'''Returns the target for asession to connect to this server.
      '''
      if self._is_local:
        return ''
      return super().target

  return HybridBackendServer


Server = wraps_server(server_lib.Server)


def MonitoredTrainingSession(  # pylint: disable=invalid-name
    master='',
    is_chief=True,
    checkpoint_dir=None,
    **kwargs):
  r'''Creates a `MonitoredSession` for training.
  '''
  hooks = kwargs.pop('hooks', [])
  hooks.extend(Context.get().training_hooks)
  kwargs['hooks'] = hooks
  kwargs['config'] = build_session_config(kwargs.pop('config', None))
  with ops.device(device_setter()):
    return monitored_session.MonitoredTrainingSession(
        master=master,
        is_chief=is_chief,
        checkpoint_dir=checkpoint_dir,
        **kwargs)

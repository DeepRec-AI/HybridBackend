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

r'''Support for training models in hybridbackend.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hybridbackend.tensorflow.training.optimizer import wraps_optimizer
from hybridbackend.tensorflow.training.saver import replace_default_saver
from hybridbackend.tensorflow.training.saver import Saver
from hybridbackend.tensorflow.training.server import wraps_server
from hybridbackend.tensorflow.training.server import Server
from hybridbackend.tensorflow.training.server import MonitoredTrainingSession
from hybridbackend.tensorflow.training.server_lib import device_setter
from hybridbackend.tensorflow.training.step_stat_hook import StepStatHook

from tensorflow.python.training import training as _training
for c in _training.__dict__.values():
  if (isinstance(c, type)
      and issubclass(c, _training.Optimizer)
      and c not in (_training.Optimizer, _training.SyncReplicasOptimizer)):
    globals()[c.__name__] = wraps_optimizer(c)

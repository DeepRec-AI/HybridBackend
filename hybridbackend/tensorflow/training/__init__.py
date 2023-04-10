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

from tensorflow.python.training import training as _training

from hybridbackend.tensorflow.framework.context import Context as _ctx
from hybridbackend.tensorflow.framework.ops import ModeKeys as _mode_keys
from hybridbackend.tensorflow.framework.rewriting import GraphRewriting
from hybridbackend.tensorflow.framework.rewriting import SessionRunRewriting
from hybridbackend.tensorflow.training.evaluation import EvaluationHook
from hybridbackend.tensorflow.training.evaluation import EvaluationSpec
from hybridbackend.tensorflow.training.hooks import Policy
from hybridbackend.tensorflow.training.hooks import StepStatHook
from hybridbackend.tensorflow.training.optimizer import SyncReplicasOptimizer
from hybridbackend.tensorflow.training.optimizer import \
  wraps_optimizer as _wraps
from hybridbackend.tensorflow.training.saved_model import export
from hybridbackend.tensorflow.training.saved_model import export_all
from hybridbackend.tensorflow.training.saver import replace_default_saver
from hybridbackend.tensorflow.training.saver import Saver
from hybridbackend.tensorflow.training.server import monitored_session
from hybridbackend.tensorflow.training.server import Server
from hybridbackend.tensorflow.training.server import target
from hybridbackend.tensorflow.training.server import wraps_server
from hybridbackend.tensorflow.training.session import \
  wraps_monitored_training_session

_ = (
  _ctx.get().options
  .register('grad_lazy_sync', False, env='HB_GRAD_LAZY_SYNC')
  .register('sharding', False)
  .register(
    'use_hierarchical_embedding_lookup', False,
    env='HB_USE_HIERARCHICAL_EMBEDDING_LOOKUP')
  .register('batch_size', -1)
  .register('model_dir', None)
  .register('keep_checkpoint_max', None)
  .register('keep_checkpoint_every_n_hours', None)
  .register('mode', _mode_keys.TRAIN))


for c in _training.__dict__.values():
  if (isinstance(c, type)
      and issubclass(c, _training.Optimizer)
      and c not in (_training.Optimizer, _training.SyncReplicasOptimizer)):
    globals()[c.__name__] = _wraps(c)

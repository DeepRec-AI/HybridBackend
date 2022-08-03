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

r'''DeepRec related classes and functions.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hybridbackend.tensorflow.framework.context import Context as _ctx
from hybridbackend.tensorflow.framework.options import DictOption as _dict

_ = (
  _ctx.get().options
  .register('paiev_enabled', _dict(True))
  .register('paiev_regularizer', _dict(None))
  .register('paiev_caching_device', _dict(None))
  .register('paiev_partitioner', _dict(None))
  .register('paiev_validate_shape', _dict(True))
  .register('paiev_custom_getter', _dict(None))
  .register('paiev_constraint', _dict(None))
  .register('paiev_steps_to_live', _dict(None))
  .register('paiev_init_data_source', _dict(None))
  .register('paiev_ev_option', _dict(None)))

# pylint: disable=ungrouped-imports
try:
  from .ev import EmbeddingBackendDeepRecEV  # pylint: disable=unused-import
except ImportError:
  pass
# pylint: enable=ungrouped-imports

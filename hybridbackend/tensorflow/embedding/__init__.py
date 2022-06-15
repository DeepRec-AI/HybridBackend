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

r'''Embedding related classes and functions.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys as _sys

from tensorflow.python.framework import dtypes as _dtypes

from hybridbackend.tensorflow.embedding.backend import \
  EmbeddingBackend as Backend
from hybridbackend.tensorflow.embedding.default import EmbeddingBackendDefault
from hybridbackend.tensorflow.framework.context import Context as _ctx
from hybridbackend.tensorflow.framework.options import DictOption as _dict

# pylint: disable=ungrouped-imports
try:
  from hybridbackend.tensorflow.embedding.paiev.backend import \
    EmbeddingBackendPAIEV  # pylint: disable=unused-import
except ImportError:
  pass
# pylint: enable=ungrouped-imports


_ = (
  _ctx.get().options
  .register('emb_backend', 'DEFAULT', env='HB_EMB_BACKEND')
  .register('emb_buffer_size', 0, env='HB_EMB_BUFFER_SIZE')
  .register('emb_buffer_load_factor', 0.5, env='HB_EMB_BUFFER_LOAD_FACTOR')
  .register('emb_num_groups', 1, env='HB_EMB_NUM_GROUPS')
  .register('emb_wire_dtype', _dict(_dtypes.float32), env='HB_EMB_WIRE_DTYPE')
  .register('emb_enable_concat', True)
  .register('emb_device', _dict(''))
  .register('emb_input_device', _dict(''))
  .register('emb_dtype', _dict(_dtypes.float32))
  .register('emb_input_dtype', _dict(_dtypes.int64))
  .register('emb_unique', _dict(False))
  .register('emb_pad', _dict(False))
  .register('emb_num_buckets_max', _dict(_sys.maxsize))
  .register('emb_segment_rank', _dict(0)))

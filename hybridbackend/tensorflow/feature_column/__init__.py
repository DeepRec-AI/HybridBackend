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

r'''Feature columns for fully sharded training.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys as _sys

from tensorflow.python.feature_column.feature_column_v2 import embedding_column
from tensorflow.python.feature_column.feature_column_v2 import \
  shared_embedding_columns
from tensorflow.python.framework import dtypes as _dtypes

from hybridbackend.tensorflow.feature_column.dense_features import \
  DenseFeatures
from hybridbackend.tensorflow.feature_column.embedding_backend import \
  EmbeddingBackend
from hybridbackend.tensorflow.feature_column.embedding_backend_default import \
  EmbeddingBackendDefault
from hybridbackend.tensorflow.feature_column.feature_column import \
  raw_categorical_column
from hybridbackend.tensorflow.framework.context import Context as _ctx
from hybridbackend.tensorflow.framework.options import SelectorOption as _opt

_ = (
  _ctx.get().options
  .register('emb_backend', 'DEFAULT', env='HB_EMB_BACKEND')
  .register('emb_num_groups', 1, env='HB_EMB_NUM_GROUPS')
  .register('emb_wire_dtype', _opt(_dtypes.float32), env='HB_EMB_WIRE_DTYPE')
  .register('emb_enable_concat', True)
  .register('emb_device', _opt(''))
  .register('emb_input_device', _opt(''))
  .register('emb_dtype', _opt(_dtypes.float32))
  .register('emb_input_dtype', _opt(_dtypes.int64))
  .register('emb_unique', _opt(False))
  .register('emb_num_buckets_max', _opt(_sys.maxsize))
  .register('emb_segment_rank', _opt(0)))

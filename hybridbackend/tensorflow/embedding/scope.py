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

r'''Scopes for further embedding optimization.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os

from tensorflow.python.framework import ops


@contextlib.contextmanager
def embedding_scope(fused=True):
  r'''Context manager to optimize embedding lookups.
  '''
  scope_name = 'hb_embeddings__'
  if fused:
    scope_name = os.getenv(
      'HB_EMBEDDING_SCOPE_FUSION_NAME',
      'hb_fused_embeddings__')
  with ops.name_scope(scope_name):
    yield

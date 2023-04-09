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

r'''Support for various embedding backends.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=ungrouped-imports
try:
  from .deeprecev import \
    ShardedEmbeddingWeightsRewritingForDeepRecEV as _patch_ev
  from .variables import \
    ShardedEmbeddingWeightsRewritingForVariables as _patch_var
except:  # pylint: disable=bare-except
  pass
# pylint: enable=ungrouped-imports

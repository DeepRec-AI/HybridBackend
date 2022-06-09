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

r'''Operations in HybridBackend.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hybridbackend.tensorflow.framework.context import Context as _ctx

_ = (
  _ctx.get().options
  .register(
    'op_segment_reduction_sort_ids', False,
    env='HB_OP_SEGMENT_REDUCTION_SORT_IDS'))

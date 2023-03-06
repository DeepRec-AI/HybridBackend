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

r'''Dataset that reads tabular data.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=ungrouped-imports
try:
  from hybridbackend.tensorflow.data.tabular.dataset_v2 import \
    TabularDatasetV2 as Dataset
  Dataset.__module__ = __name__
  Dataset.__name__ = 'TabularDataset'
except ImportError:
  from hybridbackend.tensorflow.data.tabular.dataset_v1 import \
    TabularDatasetV1 as Dataset
  Dataset.__module__ = __name__
  Dataset.__name__ = 'TabularDataset'
# pylint: enable=ungrouped-imports

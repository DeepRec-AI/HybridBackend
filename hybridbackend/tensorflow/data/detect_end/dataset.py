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

r'''DetectEndDataset that reports the existence of next element.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  from tensorflow.python.data.ops.dataset_ops import DatasetV2 as _dataset  # pylint: disable=unused-import, ungrouped-imports

  from hybridbackend.tensorflow.data.detect_end.dataset_v2 import \
    DetectEndDatasetV2 as DetectEndDataset
  DetectEndDataset.__module__ = __name__
  DetectEndDataset.__name__ = 'DetectEndDataset'
except ImportError:
  from hybridbackend.tensorflow.data.detect_end.dataset_v1 import \
    DetectEndDatasetV1 as DetectEndDataset
  DetectEndDataset.__module__ = __name__
  DetectEndDataset.__name__ = 'DetectEndDataset'

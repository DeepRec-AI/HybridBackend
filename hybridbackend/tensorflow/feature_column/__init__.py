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

from hybridbackend.tensorflow.feature_column.dense_features import \
  DenseFeatures
from hybridbackend.tensorflow.feature_column.feature_column import \
  embedding_column
from hybridbackend.tensorflow.feature_column.feature_column import \
  shared_embedding_columns
from hybridbackend.tensorflow.feature_column.raw_categorical_column import \
  raw_categorical_column

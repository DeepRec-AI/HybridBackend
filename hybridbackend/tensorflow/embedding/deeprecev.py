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

r'''DeepRec EV as embedding tables.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import variable_scope as vs

from hybridbackend.tensorflow.embedding.sharding import \
  ShardedEmbeddingWeightsRewriting


class ShardedEmbeddingWeightsRewritingForDeepRecEV(
    ShardedEmbeddingWeightsRewriting):  # pylint: disable=useless-object-inheritance
  r'''Embedding lookup decorator for DeepRec EV.
  '''
  def __init__(self):
    super().__init__()
    self._prev_get_embedding_variable = None

  @property
  def isdynamic(self):
    r'''Whether embedding weights is dynamic.
    '''
    return True

  def begin(self):
    r'''Rewrites API.
    '''
    try:
      self._prev_get_embedding_variable = (
        vs.VariableScope.get_embedding_variable)  # pylint: disable=protected-access
      vs.VariableScope.get_embedding_variable = (  # pylint: disable=protected-access
        self.wraps_build_embedding_weights(self._prev_get_embedding_variable))
    except:  # pylint: disable=bare-except
      pass

  def end(self):
    r'''Revert API rewriting.
    '''
    try:
      vs.VariableScope.get_embedding_variable = (  # pylint: disable=protected-access
        self._prev_get_embedding_variable)
    except:  # pylint: disable=bare-except
      pass


ShardedEmbeddingWeightsRewriting.register(
  ShardedEmbeddingWeightsRewritingForDeepRecEV)

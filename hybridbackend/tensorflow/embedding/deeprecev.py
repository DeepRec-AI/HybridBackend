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
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

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

  def build_unsharded_weights(self, fn, *args, **kwargs):
    r'''Build unsharded embedding weights.
    '''
    return fn(*args, **kwargs)

  def build_sharded_weights(
      self, shard, num_shards, shard_collections, fn, *args, **kwargs):
    r'''Build sharded embedding weights.
    '''
    kwargs['collections'] = shard_collections
    embedding_dim = kwargs.get('shape', None)
    if embedding_dim is None:
      raise ValueError('missing embedding_dim for tf.get_embedding_variable')
    var_scope, var_store, name, *next_args = args
    name = f'{name}/part_{shard}' if name else f'part_{shard}'
    embedding_weights = fn(
      var_scope, var_store, name, *next_args, **kwargs)
    full_name = embedding_weights.name.split(':')[0]
    full_name = full_name[:full_name.rfind('/part')]
    if hasattr(embedding_weights, '_set_save_slice_info'):
      embedding_weights._set_save_slice_info(  # pylint: disable=protected-access
        variables.Variable.SaveSliceInfo(
          full_name=full_name,
          full_shape=[num_shards, embedding_dim],
          var_offset=[shard, 0],
          var_shape=embedding_weights.shape))
    elif isinstance(embedding_weights, variables.PartitionedVariable):
      for pvar in embedding_weights:
        pvar._set_save_slice_info(  # pylint: disable=protected-access
          variables.Variable.SaveSliceInfo(
            full_name=full_name,
            full_shape=[num_shards, embedding_dim],
            var_offset=[shard, 0],
            var_shape=pvar.shape))
    else:
      logging.warning(
        f'Embedding weights {full_name} cannot be saved correctly')

    return embedding_weights

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

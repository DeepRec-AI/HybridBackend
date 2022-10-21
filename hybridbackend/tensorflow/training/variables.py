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

r'''Variable utility for training.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.backend import reset_uids as reset_keras_uids
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.layers import base
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

from hybridbackend.tensorflow.framework.rewriting import GraphRewriting
from hybridbackend.tensorflow.training.embedding import \
  EmbeddingLookupRewriting


class ReuseVariables(object):  # pylint: disable=useless-object-inheritance
  r'''Variable reusing context.
  '''
  def __call__(self, reuse):
    reset_keras_uids()
    varscope = ops.get_default_graph().get_collection_ref(('__varscope',))
    if varscope:
      varscope[0].variable_scopes_count.clear()
    vs.get_variable_scope()._reuse = reuse  # pylint: disable=protected-access


@contextlib.contextmanager
def reuse_variables(reuse=None):
  r'''Context manager that reuses variables.
  '''
  try:
    fn = ReuseVariables()
    prev_reuse = vs.get_variable_scope()._reuse  # pylint: disable=protected-access
    if reuse is not None:
      fn(reuse)
    yield fn
  finally:
    vs.get_variable_scope()._reuse = prev_reuse  # pylint: disable=protected-access


@contextlib.contextmanager
def disable_variable_update():
  r'''Context manager that disable update in state_ops's assign operations
  '''
  try:
    def wraps_assign(assign_fn):  # pylint: disable=unused-argument
      r'''Disable the assign op
      '''
      def wrapped_assign(
          ref, value, validate_shape=None, use_locking=None, name=None):  # pylint: disable=unused-argument
        return value
      return wrapped_assign

    def wraps_assign_sub(assign_sub_fn):  # pylint: disable=unused-argument
      r'''Disable the assign_sub op
      '''
      def wrapped_assign_sub(ref, value, use_locking=None, name=None):  # pylint: disable=unused-argument
        return math_ops.subtract(ref, value)
      return wrapped_assign_sub

    def wraps_assign_add(assign_add_fn):  # pylint: disable=unused-argument
      r'''Disable the assign_add op
      '''
      def wrapped_assign_add(ref, value, use_locking=None, name=None):  # pylint: disable=unused-argument
        return math_ops.add(ref, value)
      return wrapped_assign_add

    prev_assign = state_ops.assign
    state_ops.assign = wraps_assign(prev_assign)
    prev_assign_sub = state_ops.assign_sub
    state_ops.assign_sub = wraps_assign_sub(prev_assign_sub)
    prev_assign_add = state_ops.assign_add
    state_ops.assign_add = wraps_assign_add(prev_assign_add)

    yield

  finally:
    state_ops.assign = prev_assign
    state_ops.assign_sub = prev_assign_sub
    state_ops.assign_add = prev_assign_add


class LayerRewriting(GraphRewriting):
  r'''Rewriting layers.
  '''
  def __init__(self):
    super().__init__()
    self._prev_add_weight = None

  def wraps_add_weight(self, fn):
    def wrapped_add_weight(layer, name, shape, **kwargs):
      kwargs['getter'] = vs.get_variable
      if isinstance(layer, base.Layer):
        return fn(layer, name, shape, **kwargs)
      with vs.variable_scope(layer._name):  # pylint: disable=protected-access
        return fn(layer, name, shape, **kwargs)
    return wrapped_add_weight

  def begin(self):
    r'''Rewrites API.
    '''
    self._prev_add_weight = base_layer.Layer.add_weight
    base_layer.Layer.add_weight = self.wraps_add_weight(self._prev_add_weight)

  def end(self):
    r'''Revert API rewriting.
    '''
    base_layer.Layer.add_weight = self._prev_add_weight


GraphRewriting.register(LayerRewriting)


class EmbeddingLookupRewritingForVariables(EmbeddingLookupRewriting):
  r'''Embedding lookup rewriting for variables.
  '''
  def __init__(self):
    super().__init__()
    self._prev_lookup = None
    self._prev_get_variable = None

  def build_sharded_ids(self, ids):
    r'''Shard IDs to lookup sharded embedding weights.
    '''
    return ids // self.num_shards

  def build_sharded_weights(self, shard, num_shards, fn, name, *args, **kwargs):
    r'''Build sharded embedding weights.
    '''
    shape = kwargs.get('shape', None)
    shape = tensor_shape.as_shape(shape)
    if shape.dims is None:
      return fn(name, *args, **kwargs)

    shape = shape.as_list()
    if len(shape) != 2:
      return fn(name, *args, **kwargs)

    bucket_size, dimension_size = shape
    sharded_bucket_size = bucket_size // num_shards
    if shard < bucket_size % num_shards:
      sharded_bucket_size += 1
    shape = (sharded_bucket_size, dimension_size)
    kwargs['shape'] = shape

    embedding_weights = fn(f'{name}/part_{shard}', *args, **kwargs)
    bucket_offset = (bucket_size // num_shards) * shard
    remained_buckets = bucket_size % num_shards
    if shard < remained_buckets:
      bucket_offset += shard
    else:
      bucket_offset += remained_buckets
    if hasattr(embedding_weights, '_set_save_slice_info'):
      embedding_weights._set_save_slice_info(  # pylint: disable=protected-access
        variables.Variable.SaveSliceInfo(
          full_name=name,
          full_shape=[bucket_size, dimension_size],
          var_offset=[bucket_offset, 0],
          var_shape=shape))
    elif isinstance(embedding_weights, variables.PartitionedVariable):
      for embedding_partition in embedding_weights:
        poffset = embedding_partition._get_save_slice_info().var_offset  # pylint: disable=protected-access
        embedding_partition._set_save_slice_info(  # pylint: disable=protected-access
          variables.Variable.SaveSliceInfo(
            full_name=name,
            full_shape=[bucket_size, dimension_size],
            var_offset=[bucket_offset + poffset[0], poffset[1]],
            var_shape=embedding_partition.shape))
    else:
      logging.warning(f'Embedding weights {name} cannot be saved correctly')

    return embedding_weights

  def begin(self):
    r'''Rewrites API.
    '''
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    self._prev_lookup = embedding_ops.embedding_lookup
    embedding_ops.embedding_lookup = self.wraps_embedding_lookup(
      embedding_ops.embedding_lookup)
    tf.nn.embedding_lookup = self.wraps_embedding_lookup(
      embedding_ops.embedding_lookup)

    self._prev_get_variable = vs.get_variable
    vs.get_variable = self.wraps_build_embedding_weights(
      self._prev_get_variable)
    tf.get_variable = self.wraps_build_embedding_weights(
      self._prev_get_variable)

  def end(self):
    r'''Revert API rewriting.
    '''
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    vs.get_variable = self._prev_get_variable
    tf.get_variable = self._prev_get_variable
    embedding_ops.embedding_lookup = self._prev_lookup
    tf.nn.embedding_lookup = self._prev_lookup


EmbeddingLookupRewriting.register(EmbeddingLookupRewritingForVariables)

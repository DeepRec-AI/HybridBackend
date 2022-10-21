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

r'''DeepRec EV backend of embedding tables.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

from hybridbackend.tensorflow.feature_column.backend import EmbeddingBackend
from hybridbackend.tensorflow.feature_column.default import \
  EmbeddingBackendDefault
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.training.embedding import \
  EmbeddingLookupRewriting


class EmbeddingBackendDeepRecEV(EmbeddingBackendDefault):  # pylint: disable=useless-object-inheritance
  r'''Embedding backend for EV in DeepRec.
  '''
  NAME = 'PAIEV'

  def enabled(self, column):
    r'''Enable EV for specific columns.
    '''
    enabled = (
      Context.get().options.paiev_enabled[column])
    if not enabled and self.num_buckets(column) is None:
      logging.warning(
        'Not possible to disable EV for column {column} with `None` buckets.')
      return True
    return enabled

  def sharded(self, column):
    r'''Whether the column should be sharded.
    '''
    if Context.get().world_size <= 1:
      return False
    if not self.enable_sharding:
      return False
    return True

  def device(self, column):
    r'''Device of the column weights.
    '''
    if not self.enabled(column):
      return super().device(column)

    emb_device = super().device(column)
    if 'cpu' not in emb_device.lower():
      logging.info('EV only supports CPU as embedding device')
      return '/cpu:0'
    return emb_device

  def build(
      self,
      column,
      name,
      shape,
      dtype=None,
      trainable=True,
      use_resource=True,
      initializer=None,
      collections=None,
      layer=None):
    r'''Creates the embedding lookup weight.
    '''
    if not self.enabled(column):
      if len(shape) < 2:
        raise ValueError(
          'Non PAIEV column shall have a shape of at least 2 ranks')
      return super().build(
        column, name, shape,
        dtype=dtype,
        trainable=trainable,
        use_resource=use_resource,
        initializer=initializer,
        collections=collections,
        layer=layer)

    self._input_dtype = self.input_dtype(column)
    options = Context.get().options
    regularizer = options.paiev_regularizer[column]
    caching_device = options.paiev_caching_device[column]
    partitioner = options.paiev_partitioner[column]
    validate_shape = options.paiev_validate_shape[column]
    custom_getter = options.paiev_custom_getter[column]
    constraint = options.paiev_constraint[column]
    steps_to_live = options.paiev_steps_to_live[column]
    init_data_source = options.paiev_init_data_source[column]
    ev_option = options.paiev_ev_option[column]
    if ev_option is None:
      ev_option = variables.EmbeddingVariableOption()

    var = vs.get_embedding_variable(
      name,
      shape[-1],
      key_dtype=self._input_dtype,
      value_dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=trainable,
      collections=collections,
      caching_device=caching_device,
      partitioner=partitioner,
      validate_shape=validate_shape,
      custom_getter=custom_getter,
      constraint=constraint,
      steps_to_live=steps_to_live,
      init_data_source=init_data_source,
      ev_option=ev_option)
    if hasattr(var, '_set_save_slice_info'):
      var._set_save_slice_info(  # pylint: disable=protected-access
        variables.Variable.SaveSliceInfo(
          full_name=name,
          full_shape=[Context.get().world_size, self.dimension(column)],
          var_offset=[Context.get().rank, 0],
          var_shape=shape))
    elif isinstance(var, variables.PartitionedVariable):
      for pvar in var:
        pvar._set_save_slice_info(  # pylint: disable=protected-access
          variables.Variable.SaveSliceInfo(
            full_name=name,
            full_shape=[Context.get().world_size, self.dimension(column)],
            var_offset=[Context.get().rank, 0],
            var_shape=pvar.shape))
    else:
      logging.warning(f'Restoring EV without elasticity: {var}')
    return var

  def lookup(self, column, weight, inputs, sharded=False, buffered=False):
    r'''Lookup for embedding vectors.
    '''
    if not self.enabled(column):
      return super().lookup(
        column, weight, inputs,
        sharded=sharded,
        buffered=buffered)

    inputs = math_ops.cast(inputs, self._input_dtype)
    return embedding_ops.embedding_lookup(weight, inputs)

  def update(self, column, weight, indexed_updates):
    r'''Update embedding weight.
    '''
    if not self.enabled(column):
      return super().update(column, weight, indexed_updates)

    raise NotImplementedError

  def weight_name(self, column):
    r'''Name of the column weights.
    '''
    if self.sharded(column):
      shard = Context.get().rank
      return f'embedding_weights/part_{shard}'
    return 'embedding_weights/part_0'

  def weight_shared_name(self, column, var):
    r'''Get shared name of the column weights from an variable.
    '''
    var_name = var.name.split(':')[0]
    if self.sharded(column):
      return var_name[:var_name.rfind('/part')]
    return var_name


EmbeddingBackend.register(EmbeddingBackendDeepRecEV())


class EmbeddingLookupRewritingForDeepRecEV(EmbeddingLookupRewriting):  # pylint: disable=useless-object-inheritance
  r'''Embedding lookup decorator for DeepRec EV.
  '''
  def __init__(self):
    super().__init__()
    self._prev_lookup = None
    self._prev_get_embedding_variable = None

  def build_unsharded_weights(self, fn, name, *args, **kwargs):
    r'''Build unsharded embedding weights.
    '''
    return fn(f'{name}/part_0', *args, **kwargs)

  def build_sharded_weights(self, shard, num_shards, fn, name, *args, **kwargs):
    r'''Build sharded embedding weights.
    '''
    embedding_dim = kwargs.pop('embedding_dim', None)
    if embedding_dim is None:
      try:
        embedding_dim, = args
      except ValueError as ex:
        six.raise_from(
          ValueError('missing embedding_dim for tf.get_embedding_variable'),
          ex)
    embedding_weights = fn(f'{name}/part_{shard}', embedding_dim, **kwargs)
    if hasattr(embedding_weights, '_set_save_slice_info'):
      embedding_weights._set_save_slice_info(  # pylint: disable=protected-access
        variables.Variable.SaveSliceInfo(
          full_name=name,
          full_shape=[num_shards, embedding_dim],
          var_offset=[shard, 0],
          var_shape=embedding_weights.shape))
    elif isinstance(embedding_weights, variables.PartitionedVariable):
      for pvar in embedding_weights:
        pvar._set_save_slice_info(  # pylint: disable=protected-access
          variables.Variable.SaveSliceInfo(
            full_name=name,
            full_shape=[num_shards, embedding_dim],
            var_offset=[shard, 0],
            var_shape=pvar.shape))
    else:
      logging.warning(f'Embedding weights {name} cannot be saved correctly')

    return embedding_weights

  def begin(self):
    r'''Rewrites API.
    '''
    try:
      import tensorflow as tf  # pylint: disable=import-outside-toplevel
      self._prev_lookup = embedding_ops.embedding_lookup
      embedding_ops.embedding_lookup = self.wraps_embedding_lookup(
        embedding_ops.embedding_lookup)
      tf.nn.embedding_lookup = self.wraps_embedding_lookup(
        embedding_ops.embedding_lookup)

      self._prev_get_embedding_variable = vs.get_embedding_variable
      vs.get_embedding_variable = self.wraps_build_embedding_weights(
        self._prev_get_embedding_variable)
      tf.get_embedding_variable = self.wraps_build_embedding_weights(
        self._prev_get_embedding_variable)
    except:  # pylint: disable=bare-except
      pass

  def end(self):
    r'''Revert API rewriting.
    '''
    try:
      import tensorflow as tf  # pylint: disable=import-outside-toplevel
      vs.get_embedding_variable = self._prev_get_embedding_variable
      tf.get_embedding_variable = self._prev_get_embedding_variable

      embedding_ops.embedding_lookup = self._prev_lookup
      tf.nn.embedding_lookup = self._prev_lookup
    except:  # pylint: disable=bare-except
      pass


EmbeddingLookupRewriting.register(EmbeddingLookupRewritingForDeepRecEV)

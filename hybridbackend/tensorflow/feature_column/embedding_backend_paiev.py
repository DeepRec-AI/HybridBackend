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

r'''PAI EV backend of embedding tables.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

from hybridbackend.tensorflow.feature_column.embedding_backend import \
  EmbeddingBackend
from hybridbackend.tensorflow.feature_column.embedding_backend_default import \
  EmbeddingBackendDefault
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.options import SelectorOption


class EmbeddingBackendPAIEV(EmbeddingBackendDefault):  # pylint: disable=useless-object-inheritance
  r'''EV Embedding backend for variables.
  '''
  NAME = 'PAIEV'

  _ = (
    Context.get().options
    .register('paiev_enabled', SelectorOption(True))
    .register('paiev_regularizer', SelectorOption(None))
    .register('paiev_caching_device', SelectorOption(None))
    .register('paiev_partitioner', SelectorOption(None))
    .register('paiev_validate_shape', SelectorOption(True))
    .register('paiev_custom_getter', SelectorOption(None))
    .register('paiev_constraint', SelectorOption(None))
    .register('paiev_steps_to_live', SelectorOption(None))
    .register('paiev_init_data_source', SelectorOption(None))
    .register('paiev_ev_option', SelectorOption(None)))

  def enabled(self, column):
    r'''Enable PAIEV for specific columns.
    '''
    enabled = (
      Context.get().options.paiev_enabled[column.categorical_column.name])
    if not enabled and self.num_buckets(column) is None:
      logging.warning(
        'Not possible to disable PAIEV for column '
        f'{column.categorical_column.name} with `None` buckets.')
      return True
    return enabled

  def sharded(self, column):
    r'''Whether the column should be sharded.
    '''
    if Context.get().world_size <= 1:
      return False
    if not self.enable_sharding:
      return False
    batch_size = Context.get().options.batch_size
    if batch_size < 0 or self.num_buckets(column) is None:
      return True
    if batch_size < self.num_buckets(column):
      return True
    return True

  def device(self, column):
    r'''Device of the column weights.
    '''
    if not self.enabled(column):
      return super().device(column)

    emb_device = super().device(column)
    if 'cpu' not in emb_device.lower():
      logging.info('PAIEV only supports CPU as embedding device')
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
      shape = (self.num_buckets(column), shape[1])
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
    column_name = column.categorical_column.name
    regularizer = options.paiev_regularizer[column_name]
    caching_device = options.paiev_caching_device[column_name]
    partitioner = options.paiev_partitioner[column_name]
    validate_shape = options.paiev_validate_shape[column_name]
    custom_getter = options.paiev_custom_getter[column_name]
    constraint = options.paiev_constraint[column_name]
    steps_to_live = options.paiev_steps_to_live[column_name]
    init_data_source = options.paiev_init_data_source[column_name]
    ev_option = options.paiev_ev_option[column_name]
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
      logging.warning(f'Restoring PAIEV without elasticity: {var}')
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


EmbeddingBackend.register(EmbeddingBackendPAIEV())

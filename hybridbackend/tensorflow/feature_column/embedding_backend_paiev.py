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
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.feature_column.embedding_backend import \
  EmbeddingBackend
from hybridbackend.tensorflow.feature_column.embedding_backend_default import \
  EmbeddingBackendDefault


class EmbeddingBackendPAIEV(EmbeddingBackendDefault):  # pylint: disable=useless-object-inheritance
  r'''EV Embedding backend for variables.
  '''
  NAME = 'PAIEV'

  def enabled(self, column):
    r'''Enable PAIEV for specific columns.
    '''
    enabled = Context.get().param(
      'emb_backend_paiev', True,
      env='HB_EMB_BACKEND_PAIEV',
      parser=Context.parse_bool,
      select=column.categorical_column.name)
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
    batch_size = Context.get().batch_size
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
    paiev_get_var_args = {}
    paiev_get_var_args['regularizer'] = Context.get().param(
      'paiev_regularizer', None)
    paiev_get_var_args['caching_device'] = Context.get().param(
      'paiev_caching_device', None)
    paiev_get_var_args['partitioner'] = Context.get().param(
      'paiev_partitioner', None)
    paiev_get_var_args['validate_shape'] = Context.get().param(
      'paiev_validate_shape', True)
    paiev_get_var_args['custom_getter'] = Context.get().param(
      'paiev_custom_getter', None)
    paiev_get_var_args['constraint'] = Context.get().param(
      'paiev_constraint', None)
    paiev_get_var_args['steps_to_live'] = Context.get().param(
      'paiev_steps_to_live', None)
    paiev_get_var_args['init_data_source'] = Context.get().param(
      'paiev_init_data_source', None)
    paiev_get_var_args['ev_option'] = Context.get().param(
      'paiev_ev_option', variables.EmbeddingVariableOption())
    for key, val in paiev_get_var_args.items():
      if isinstance(val, dict):
        paiev_get_var_args[key] = val[column]

    var = vs.get_embedding_variable(
      name,
      shape[-1],
      key_dtype=self._input_dtype,
      value_dtype=dtype,
      initializer=initializer,
      regularizer=paiev_get_var_args['regularizer'],
      trainable=trainable,
      collections=collections,
      caching_device=paiev_get_var_args['caching_device'],
      partitioner=paiev_get_var_args['partitioner'],
      validate_shape=paiev_get_var_args['validate_shape'],
      custom_getter=paiev_get_var_args['custom_getter'],
      constraint=paiev_get_var_args['constraint'],
      steps_to_live=paiev_get_var_args['steps_to_live'],
      init_data_source=paiev_get_var_args['init_data_source'],
      ev_option=paiev_get_var_args['ev_option'])
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


EmbeddingBackend.register(EmbeddingBackendPAIEV())

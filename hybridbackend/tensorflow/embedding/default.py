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

r'''Default backend of embedding tables.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils

from hybridbackend.tensorflow.embedding.backend import EmbeddingBackend
from hybridbackend.tensorflow.framework.context import Context


class EmbeddingBackendDefault(EmbeddingBackend):
  r'''Embedding backend for variables.
  '''
  NAME = 'DEFAULT'

  def _get_variable(
      self,
      name,
      shape=None,
      dtype=dtypes.float32,
      initializer=None,
      trainable=None,
      constraint=None,
      use_resource=None,
      collections=None,
      synchronization=variables.VariableSynchronization.AUTO,
      aggregation=variables.VariableAggregation.NONE,
      partitioner=None):  # pylint: disable=unused-argument
    r'''Gets an existing variable with these parameters or create a new one.
    '''
    return vs.get_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      trainable=trainable,
      constraint=constraint,
      use_resource=use_resource,
      collections=collections,
      synchronization=synchronization,
      aggregation=aggregation,
      partitioner=partitioner)

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
    del use_resource
    num_buckets_max = Context.get().options.emb_num_buckets_max[
      column.categorical_column.name]
    num_buckets, dim = shape
    if num_buckets is None or num_buckets >= num_buckets_max:
      raise ValueError(
        f'num_buckets of categorical_column {column.categorical_column.name} '
        'must be set explicitly')
    if self.sharded(column):
      shard = Context.get().rank
      num_shards = Context.get().world_size
      sharded_bucket_size = num_buckets // num_shards
      if shard < num_buckets % num_shards:
        sharded_bucket_size += 1
      shape = (sharded_bucket_size, dim)
    if layer is None:
      var = vs.get_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        trainable=trainable,
        use_resource=False,
        collections=collections)
    else:
      var = layer.add_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        trainable=trainable,
        use_resource=False,
        getter=self._get_variable,
        collections=collections)
    if self.sharded(column):
      num_buckets = getattr(column.categorical_column, 'num_buckets',
                            column.categorical_column._num_buckets)  # pylint: disable=protected-access
      num_shards = Context.get().world_size
      shard = Context.get().rank
      bucket_offset = (num_buckets // num_shards) * shard
      remained_buckets = num_buckets % num_shards
      if shard < remained_buckets:
        bucket_offset += shard
      else:
        bucket_offset += remained_buckets
      full_name = self.weight_shared_name(column, var)
      if hasattr(var, '_set_save_slice_info'):
        var._set_save_slice_info(  # pylint: disable=protected-access
          variables.Variable.SaveSliceInfo(
            full_name=full_name,
            full_shape=[num_buckets, self.dimension(column)],
            var_offset=[bucket_offset, 0],
            var_shape=shape))
      elif isinstance(var, variables.PartitionedVariable):
        for pvar in var:
          poffset = pvar._get_save_slice_info().var_offset  # pylint: disable=protected-access
          pvar._set_save_slice_info(  # pylint: disable=protected-access
            variables.Variable.SaveSliceInfo(
              full_name=full_name,
              full_shape=[num_buckets, self.dimension(column)],
              var_offset=[bucket_offset + poffset[0], poffset[1]],
              var_shape=pvar.shape))
      else:
        logging.warning(f'Saving variable without elasticity: {var}')
      return var
    return var

  def init_from_checkpoint(
      self, column, ckpt_dir_or_file, tensor_name_in_ckpt, to_restore):
    r'''Replaces initializers of embedding weights to load from checkpoints.
    '''
    del column
    if isinstance(to_restore, variables.PartitionedVariable):
      to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
    checkpoint_utils.init_from_checkpoint(
      ckpt_dir_or_file, {tensor_name_in_ckpt: to_restore})

  def lookup(self, column, weight, inputs, sharded=False, buffered=False):
    r'''Lookup for embedding vectors.
    '''
    del column
    del buffered
    if sharded:
      num_shards = Context.get().world_size
      return array_ops.gather(weight, inputs // num_shards)
    return array_ops.gather(weight, inputs)

  def update(self, column, weight, indexed_updates):
    r'''Update embedding weight.
    '''
    del column
    return weight.scatter_update(indexed_updates)


EmbeddingBackend.register(EmbeddingBackendDefault())

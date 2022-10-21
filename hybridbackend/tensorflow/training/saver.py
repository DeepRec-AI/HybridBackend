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

r'''Save and restore replicated and sharded variables.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver
from tensorflow.python.util import nest

try:
  from tensorflow.python.training.saving.saveable_object_util import \
    op_list_to_dict
except ImportError:
  op_list_to_dict = saver.BaseSaverBuilder.OpListToDict

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import GraphKeys
from hybridbackend.tensorflow.framework.ops import ModeKeys
from hybridbackend.tensorflow.framework.rewriting import GraphRewriting
from hybridbackend.tensorflow.framework.rewriting import SessionRunRewriting


class HybridBackendSaverBuilderBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of sharded saver builders.
  '''


def wraps_saver_builder(cls):
  r'''Wraps a saver builder to support hybrid parallelism.
  '''
  if issubclass(cls, HybridBackendSaverBuilderBase):
    return cls

  class HybridBackendSaverBuilder(cls, HybridBackendSaverBuilderBase):
    r'''Wrapped SaverBuilder with support for hybrid parallelism.
    '''
    def __init__(self, *args, **kwargs):
      name = kwargs.pop('name', 'hybrid_backend_saver_builder')
      self._restoreable_saveables = None
      self._rank = Context.get().rank
      self._world_size = Context.get().world_size
      super().__init__(*args, **kwargs)
      with ops.device(Context.get().devices[0]):
        with ops.device('/cpu:0'):
          self._local_barrier = data_flow_ops.Barrier(
            [dtypes.bool],
            shared_name=f'{name}_local_barrier')
          self._global_barrier = data_flow_ops.Barrier(
            [dtypes.bool],
            shared_name=f'{name}_global_barrier')

    @property
    def rank(self):
      return self._rank

    @property
    def world_size(self):
      return self._world_size

    def _AddShardedSaveOpsForV2(self, checkpoint_prefix, per_device):
      r'''Add ops to save the params in parallel, for the V2 format.

      Args:
        checkpoint_prefix: scalar String Tensor.  Interpreted *NOT AS A
          FILENAME*, but as a prefix of a V2 checkpoint;
        per_device: A list of (device, BaseSaverBuilder.VarToSave) pairs, as
          returned by _GroupByDevices().

      Returns:
        An op to save the variables, which, when evaluated, returns the prefix
          "<user-fed prefix>" only and does not include the sharded spec suffix.
      '''
      if self._world_size <= 1:
        return super()._AddShardedSaveOpsForV2(checkpoint_prefix, per_device)

      # Filter sharded saveables.
      sharded_saveables = []
      if self._rank != 0:
        sharded_saveables += ops.get_collection_ref(
          GraphKeys.SHARDED_VARIABLES)
        sharded_saveables += ops.get_collection_ref(
          GraphKeys.SHARDED_RESOURCES)
        sharded_saveables = [v for v in sharded_saveables if v is not None]
        sharded_saveables = op_list_to_dict(sharded_saveables).values()
        sharded_saveables = nest.flatten(sharded_saveables)

      # Save local partitions.
      checkpoint_uuid = uuid.uuid4()
      checkpoint_prefix_suffix = f'_temp_{checkpoint_uuid.hex}/part'
      tmp_checkpoint_prefix = string_ops.string_join(
        [checkpoint_prefix, checkpoint_prefix_suffix])

      num_shards = len(per_device)
      num_shards_tensor = constant_op.constant(num_shards)
      local_done = constant_op.constant([True], dtype=dtypes.bool)
      global_done = constant_op.constant(
        [True for _ in range(self._world_size - 1)], dtype=dtypes.bool)
      save_ops = []
      filenames = []
      last_device = None
      empty_filename = ''

      for shard, (device, saveables) in enumerate(per_device):
        # Only sharded saveables need to save for non-chief workers
        if self._rank != 0:
          saveables = [
            s for s in saveables
            if s.op in sharded_saveables or s in sharded_saveables]
        last_device = device
        with ops.device(device):
          with ops.device('/cpu:0'):
            filename = empty_filename
            if saveables:
              filename = self.sharded_filename(
                tmp_checkpoint_prefix, shard, num_shards_tensor)
              save_ops.append(self._AddSaveOps(filename, saveables))
            filenames.append(filename)

      with ops.control_dependencies([x.op for x in save_ops]):
        with ops.device(last_device):
          with ops.device('/cpu:0'):
            notify_local_done = [
              self._local_barrier.insert_many(0, keys=[f], values=local_done)
              for f in filenames]
            _, ready_filenames, _ = self._local_barrier.take_many(
              self._world_size * len(filenames))
            notify_global_done = self._global_barrier.insert_many(
              0,
              keys=[str(i) for i in range(self._world_size - 1)],
              values=global_done)
            _, ready_ranks, _ = self._global_barrier.take_many(1)

            if self._rank == 0:
              ready_filenames_mask = math_ops.logical_not(
                string_ops.regex_full_match(ready_filenames, empty_filename))
              ready_filenames = array_ops.boolean_mask(
                ready_filenames, ready_filenames_mask)
              with ops.control_dependencies(notify_local_done):
                with ops.control_dependencies([ready_filenames]):
                  merge_files = gen_io_ops.merge_v2_checkpoints(
                    ready_filenames, checkpoint_prefix, delete_old_dirs=True)
                with ops.control_dependencies([merge_files]):
                  with ops.control_dependencies([notify_global_done]):
                    return array_ops.identity(checkpoint_prefix)
            with ops.control_dependencies(notify_local_done):
              with ops.control_dependencies([ready_ranks]):
                return array_ops.identity(checkpoint_prefix)

    def _AddShardedRestoreOps(
        self, filename_tensor, per_device, restore_sequentially, reshape):
      r'''Add Ops to restore variables from multiple devices.

      Args:
        filename_tensor: Tensor for the path of the file to load.
        per_device: A list of (device, SaveableObject) pairs, as returned by
          _GroupByDevices().
        restore_sequentially: True if we want to restore variables sequentially
          within a shard.
        reshape: True if we want to reshape loaded tensors to the shape of the
          corresponding variable.

      Returns:
        An Operation that restores the variables.
      '''
      model_dir = Context.get().options.model_dir
      if model_dir is not None:
        latest_path = checkpoint_management.latest_checkpoint(model_dir)  # pylint: disable=protected-access
        if latest_path:
          self._restoreable_saveables, _ = zip(
            *checkpoint_utils.list_variables(model_dir))
      return super()._AddShardedRestoreOps(
        filename_tensor, per_device, restore_sequentially, reshape)

    def restore_op(self, filename_tensor, saveable, preferred_shard):
      r'''Create ops to restore 'saveable'.
      '''
      if (self._restoreable_saveables is not None
          and saveable.name not in self._restoreable_saveables
          and hasattr(saveable, 'initializer')):
        return saveable.initializer.outputs
      return super().restore_op(filename_tensor, saveable, preferred_shard)

  return HybridBackendSaverBuilder


SaverBuilder = wraps_saver_builder(saver.BulkSaverBuilder)


class HybridBackendSaverBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of sharded savers.
  '''


def wraps_saver(cls):
  r'''Wraps a saver to support hybrid parallelism.
  '''
  if issubclass(cls, HybridBackendSaverBase):
    return cls

  class HybridBackendSaver(cls, HybridBackendSaverBase):
    r'''SaverBuilder with support for hybrid parallelism.
    '''
    def __init__(self, *args, **kwargs):
      self._rank = Context.get().rank
      self._world_size = Context.get().world_size
      kwargs['sharded'] = True
      kwargs['allow_empty'] = True
      with ops.device('/cpu:0'):
        super().__init__(*args, **kwargs)

    @property
    def rank(self):
      return self._rank

    @property
    def world_size(self):
      return self._world_size

    def _build(self, *args, **kwargs):
      r'''Builds saver_def.
      '''
      if self._world_size <= 1:
        super()._build(*args, **kwargs)
        return

      if self._builder is None:
        orig_saver_builder = saver.BulkSaverBuilder
        saver.BulkSaverBuilder = wraps_saver_builder(saver.BulkSaverBuilder)
        super()._build(*args, **kwargs)
        saver.BulkSaverBuilder = orig_saver_builder
      else:
        if not isinstance(self._builder, HybridBackendSaverBuilderBase):
          raise ValueError(
            '`SaverBuilder` must decorated by `wraps_saver_builder`')
        super()._build(*args, **kwargs)

    def save(self, *args, **kwargs):
      r'''Saves sharded variables.
      '''
      if self._world_size <= 1:
        super().save(*args, **kwargs)
        return

      write_meta_graph = (
        kwargs.pop('write_meta_graph', True)
        and self._rank == 0)
      kwargs['write_meta_graph'] = write_meta_graph
      write_state = kwargs.pop('write_state', True) and self._rank == 0
      kwargs['write_state'] = write_state
      super().save(*args, **kwargs)

    def export_meta_graph(self, filename=None, **kwargs):
      if self._rank == 0:
        return super().export_meta_graph(filename=filename, **kwargs)
      return None

  return HybridBackendSaver


Saver = wraps_saver(saver.Saver)


def replace_default_saver():
  r'''Try to replace default saver to HybridBackendSaver.
  '''
  rank = Context.get().rank
  savers = ops.get_collection_ref(ops.GraphKeys.SAVERS)

  if not savers:
    default_saver = Saver()
    ops.add_to_collection(ops.GraphKeys.SAVERS, default_saver)
    return
  if len(savers) > 1:
    raise ValueError(f'Multiple items found in collection SAVERS: {savers}')

  default_saver = savers[0]
  if isinstance(default_saver, HybridBackendSaverBase):
    return

  if not default_saver._sharded:  # pylint: disable=protected-access
    raise ValueError('Default saver must be sharded')
  if default_saver._builder is not None:  # pylint: disable=protected-access
    if not isinstance(default_saver._builder, HybridBackendSaverBuilderBase):  # pylint: disable=protected-access
      raise ValueError(
        'builder for default saver must decorated by `wraps_saver_builder`')
  else:
    def _wraps_build(build_fn):
      r'''Decorator to wrap saver build.
      '''
      def wrapped_build(self, *args, **kwargs):
        r'''Builds saver_def.
        '''
        orig_saver_builder = saver.BulkSaverBuilder
        saver.BulkSaverBuilder = wraps_saver_builder(saver.BulkSaverBuilder)
        build_fn(self, *args, **kwargs)
        saver.BulkSaverBuilder = orig_saver_builder
      return wrapped_build
    default_saver._build = _wraps_build(default_saver._build)  # pylint: disable=protected-access

  def _wraps_save(save_fn):
    def wrapped_save(self, *args, **kwargs):
      r'''Saves sharded variables.
      '''
      write_meta_graph = kwargs.pop('write_meta_graph', True) and rank == 0
      kwargs['write_meta_graph'] = write_meta_graph
      write_state = kwargs.pop('write_state', True) and rank == 0
      kwargs['write_state'] = write_state
      save_fn(self, *args, **kwargs)
    return wrapped_save
  default_saver.save = _wraps_save(default_saver.save)


class SaverRewriting(GraphRewriting):
  r'''Rewriting savers.
  '''
  def __init__(self):
    super().__init__()
    self._prev_add_weight = None

  def wraps_saver_init(self, fn):
    r'''Wraps saver init function.
    '''
    def wrapped_saver_init(cls, *args, **kwargs):
      keep_checkpoint_max = Context.get().options.keep_checkpoint_max
      keep_checkpoint_every_n_hours = \
        Context.get().options.keep_checkpoint_every_n_hours
      if keep_checkpoint_max is not None:
        kwargs['max_to_keep'] = keep_checkpoint_max
      if keep_checkpoint_every_n_hours is not None:
        kwargs['keep_checkpoint_every_n_hours'] = \
          keep_checkpoint_every_n_hours
      kwargs['save_relative_paths'] = True
      fn(cls, *args, **kwargs)
    return wrapped_saver_init

  def begin(self):
    r'''Rewrites API.
    '''
    self._prev_saver_init = saver.Saver.__init__
    saver.Saver.__init__ = self.wraps_saver_init(self._prev_saver_init)

  def end(self):
    r'''Revert API rewriting.
    '''
    saver.Saver.__init__ = self._prev_saver_init


GraphRewriting.register(SaverRewriting)


class DefaultSaverRewriting(SessionRunRewriting):
  r'''A SessionRunHook replaces default saver.
  '''
  def begin(self):
    r''' initialize replica variables and enable synchronous dataset wrapper
    '''
    replace_default_saver()


SessionRunRewriting.register(
  DefaultSaverRewriting, [ModeKeys.TRAIN])

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

r'''Functions for data access.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import session_run_hook

from hybridbackend.tensorflow.data.detect_end.dataset import DetectEndDataset
from hybridbackend.tensorflow.data.prefetch.ops import Iterator
from hybridbackend.tensorflow.distribute.communicator import CollectiveOps
from hybridbackend.tensorflow.distribute.communicator_pool import \
  CommunicatorPool
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import ModeKeys
from hybridbackend.tensorflow.framework.ops import MultiValues
from hybridbackend.tensorflow.framework.rewriting import GraphRewriting
from hybridbackend.tensorflow.framework.rewriting import SessionRunRewriting


def make_one_shot_iterator(ds):
  r'''Wrapper of make_one_shot_iterator.

  Args:
    ds: a `tf.data.Dataset`
  '''
  with ops.device('/cpu:0'):
    if hasattr(dataset_ops, 'make_one_shot_iterator'):
      return dataset_ops.make_one_shot_iterator(ds)
    return ds.make_one_shot_iterator()


def make_initializable_iterator(ds):
  r'''Wrapper of make_initializable_iterator.

  Args:
    ds: a `tf.data.Dataset`
  '''
  with ops.device('/cpu:0'):
    if hasattr(dataset_ops, 'make_initializable_iterator'):
      return dataset_ops.make_initializable_iterator(ds)
    return ds.make_initializable_iterator()


class HybridBackendItertorBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of iterator wrapper.
  '''


class IteratorRewriting(GraphRewriting):
  r'''Rewriting iterators.
  '''
  def __init__(self):
    super().__init__()
    self._prev_make_one_shot_iterator = None
    self._prev_make_initializable_iterator = None
    self._prev_keras_get_iterator = None
    self._prev_iterator = None

  def wraps_make_iterator(self, fn):
    r'''Wraps make_*_iterator.
    '''
    def wrapped_make_iterator(ds):
      with ops.device('/cpu:0'):
        options = Context.get().options
        if options.mode == ModeKeys.TRAIN:
          return fn(DetectEndDataset(ds))
        if options.mode == ModeKeys.EVAL:
          return fn(ds.repeat())
        return fn(ds)
    return wrapped_make_iterator

  def wraps_iterator(self, cls):
    r'''Iterator decorator to support advanced functionalities.
    '''
    if issubclass(cls, HybridBackendItertorBase):
      return cls

    class HybridBackendIterator(cls, HybridBackendItertorBase):
      r'''Class to iteratively obtain data.
      '''
      def get_next(self):
        r'''Get next batch.
        '''
        options = Context.get().options
        if options.mode != ModeKeys.TRAIN:
          return Iterator(super().get_next()).get_next()
        if options.data_batch_count > 1:
          batches = []
          should_stop = False
          for _ in range(options.data_batch_count):
            should_stop, batch = super().get_next()
            batches.append(batch)
          DataSyncRewriting.accept(should_stop)
          return MultiValues.build(batches).regroup()
        should_stop, batch = Iterator(super().get_next()).get_next()
        DataSyncRewriting.accept(should_stop)
        return batch
    return HybridBackendIterator

  def begin(self):
    r'''Rewrites API.
    '''
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    self._prev_make_one_shot_iterator = dataset_ops.make_one_shot_iterator
    dataset_ops.make_one_shot_iterator = self.wraps_make_iterator(
      self._prev_make_one_shot_iterator)
    tf.data.make_one_shot_iterator = dataset_ops.make_one_shot_iterator
    self._prev_make_initializable_iterator = \
      dataset_ops.make_initializable_iterator
    dataset_ops.make_initializable_iterator = self.wraps_make_iterator(
      self._prev_make_initializable_iterator)
    self._prev_keras_get_iterator = training_utils.get_iterator
    tf.data.make_initializable_iterator = \
      dataset_ops.make_initializable_iterator
    training_utils.get_iterator = self.wraps_make_iterator(
      self._prev_keras_get_iterator)
    self._prev_iterator = iterator_ops.Iterator
    iterator_ops.Iterator = self.wraps_iterator(self._prev_iterator)

  def end(self):
    r'''Revert API rewriting.
    '''
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    iterator_ops.Iterator = self._prev_iterator
    dataset_ops.make_one_shot_iterator = self._prev_make_one_shot_iterator
    tf.data.make_one_shot_iterator = dataset_ops.make_one_shot_iterator
    dataset_ops.make_initializable_iterator = \
      self._prev_make_initializable_iterator
    tf.data.make_initializable_iterator = \
      dataset_ops.make_initializable_iterator
    training_utils.get_iterator = self._prev_keras_get_iterator


GraphRewriting.register(IteratorRewriting)


class DataSyncRewriting(SessionRunRewriting):
  r'''Hook that synchonizes dataset reading across devices.
  '''
  @classmethod
  def accept(cls, should_stop):
    r'''Register should_stop.
    '''
    with ops.device(Context.get().devices[Context.get().rank]):
      should_stop = math_ops.cast(should_stop, dtypes.int32)
      if Context.get().options.data_sync_drop_remainder:
        should_stop_all = CommunicatorPool.get().allreduce(
          should_stop, reduce_op=CollectiveOps.MAX, trainable=False)
        return SessionRunRewriting.add_to_collection(
          DataSyncRewriting.__name__, should_stop_all)
      should_stop_all = CommunicatorPool.get().allreduce(
        should_stop, reduce_op=CollectiveOps.MIN, trainable=False)
      return SessionRunRewriting.add_to_collection(
        DataSyncRewriting.__name__, should_stop_all)

  def begin(self):
    r'''Initialize should_stop_all operation.
    '''
    should_stop_all_ops = self.get_collection(DataSyncRewriting.__name__)
    if len(should_stop_all_ops) < 1:
      self._should_stop_all = None
    elif len(should_stop_all_ops) == 1:
      self._should_stop_all = should_stop_all_ops[0]
    else:
      with ops.device(should_stop_all_ops[0].device):
        self._should_stop_all = math_ops.reduce_max(
          array_ops.concat(should_stop_all_ops, 0), 0)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    r'''Call this before sess run.
    '''
    fetches = {}
    if self._should_stop_all is not None:
      fetches['should_stop_all'] = self._should_stop_all
    return session_run_hook.SessionRunArgs(fetches)

  def after_run(self, run_context, run_values):
    r'''Call this after sess run to stop the execution.
    '''
    if ('should_stop_all' in run_values.results
        and run_values.results['should_stop_all'] > 0):
      run_context.request_stop()


SessionRunRewriting.register(
  DataSyncRewriting, [ModeKeys.TRAIN])

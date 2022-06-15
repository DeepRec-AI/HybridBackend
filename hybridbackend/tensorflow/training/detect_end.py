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

r'''Session run hook for detecting end of a dataset.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model.model_utils import mode_keys
from tensorflow.python.training import session_run_hook

from hybridbackend.tensorflow.distribute.communicator import CollectiveOps
from hybridbackend.tensorflow.distribute.communicator_pool import \
  CommunicatorPool
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.training.detect_end_dataset import \
  DetectEndDataset


class DetectEndHook(session_run_hook.SessionRunHook):
  r'''Hook that detect the end of reaching a dataset.
  '''
  def __init__(
      self,
      drop_remainder=True,
      end_marker=None):
    r'''Creates hook to detect end of datasets.

    Args:
      drop_remainder: If True, all workers would stop training and
        dump remained samples when any of them reaching the end of their
        training samples.
      name: name of the hook
    '''
    self._drop_join_remainder = drop_remainder
    self._end_marker = end_marker
    if self._end_marker is None:
      raise ValueError('Must provide an end_marker tensor to this hook')

  def _sync_end(self, reduce_op):
    r'''Synchronize end marker over all workers.
    '''
    def comm_fn(comm, inputs, inputs_deps):
      with ops.control_dependencies(inputs_deps):
        return comm.allreduce(inputs[0], reduce_op=reduce_op), None
    return comm_fn

  def begin(self):
    r'''Initialize the variables to record the end of datasets.
    '''
    if 'GPU' in Context.current_device():
      self._detect_end_comm_device = Context.current_device()
    else:
      self._detect_end_comm_device = '/gpu:0'

    with ops.device(self._detect_end_comm_device):
      end_marker = math_ops.cast(self._end_marker, dtype=dtypes.int32)
      if self._drop_join_remainder:
        reduce_op = CollectiveOps.MAX
      else:
        reduce_op = CollectiveOps.MIN
      self._end_marker_reduced = CommunicatorPool.get().call(
        self._sync_end(reduce_op),
        end_marker,
        trainable=False)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    r'''Call this before sess run.
    '''
    fetches = {}
    if self._end_marker_reduced is not None:
      fetches['end_marker'] = self._end_marker_reduced
    return session_run_hook.SessionRunArgs(fetches)

  def after_run(self, run_context, run_values):
    r'''Call this after sess run to stop the execution.
    '''
    if ('end_marker' in run_values.results
        and run_values.results['end_marker'] > 0):
      run_context.request_stop()


def _add_detect_end_hook(end_marker, drop_remainder):
  r'''create and add detect_end_hooks.
  '''
  options = Context.get().options
  if mode_keys.is_train(options.mode):
    Context.get().add_training_hook(
      DetectEndHook(
        drop_remainder=drop_remainder if drop_remainder is not None else True,
        end_marker=end_marker))
  elif mode_keys.is_eval(options.mode):
    Context.get().add_evaluation_hook(
      DetectEndHook(
        drop_remainder=drop_remainder if drop_remainder is not None else False,
        end_marker=end_marker))
  elif mode_keys.is_predict(options.mode):
    Context.get().add_prediction_hook(
      DetectEndHook(
        drop_remainder=drop_remainder if drop_remainder is not None else False,
        end_marker=end_marker))
  else:
    raise ValueError(
      f'mode must be train, eval, or infer, but it is {options.mode}')


def _wraps_iterator(cls, drop_remainder):
  r'''Iterator decorator to support advanced functionalities.
  '''
  class HybridBackendIterator(cls):
    r'''Class to iteratively obtain data.
    '''
    def get_next(self):
      r'''Get next batch.
      '''
      options = Context.get().options
      if (mode_keys.is_train(options.mode)
          or mode_keys.is_eval(options.mode)):
        end_marker, batch = super().get_next()
        _add_detect_end_hook(end_marker, drop_remainder)
        return batch

      return super().get_next()
  return HybridBackendIterator


@contextlib.contextmanager
def raises_out_of_range(ds, drop_remainder):
  r'''Context manager that raises out-of-range error for sync training.
  '''
  options = Context.get().options
  sync_end = (
    mode_keys.is_train(options.mode)
    or mode_keys.is_eval(options.mode))
  prev_iterator = None
  try:
    if sync_end:
      ds = DetectEndDataset(ds)
      prev_iterator = iterator_ops.Iterator
      iterator_ops.Iterator = _wraps_iterator(prev_iterator, drop_remainder)
    yield ds
  finally:
    if sync_end:
      iterator_ops.Iterator = prev_iterator

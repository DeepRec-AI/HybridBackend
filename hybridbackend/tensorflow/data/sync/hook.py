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

r'''SyncReplicasDataset that reports the existence of next element.

This class is compatible with Tensorflow 1.12.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import nest


class SyncReplicasDatasetHook(session_run_hook.SessionRunHook):
  r'''SessionRunHook that cooperates with SyncReplicasDataset.
  '''
  @classmethod
  def all_instances(cls):
    return ops.get_collection(cls.__name__)

  @classmethod
  def sync_all_instances(cls, features=None):
    for hook in ops.get_collection(cls.__name__):
      hook.sync(features)

  def __init__(self, world, rank, drop_remainder=True):
    r'''Initializes SyncReplicasDatasetHook.
    '''
    self._name = ops.get_default_graph().unique_name(self.__class__.__name__)
    self._world = world
    self._rank = rank
    self._drop_remainder = drop_remainder
    self._should_stop = None
    self._should_stop_count = None
    ops.add_to_collection(self.__class__.__name__, self)

  @property
  def name(self):
    return self._name

  def register(self, should_stop, values):
    r'''Register should_stop tensor and return remaining values.
    '''
    if self._should_stop is not None:
      raise ValueError('register should be done only once')

    value_list = nest.flatten(values)
    with ops.device(self._world[self._rank]), ops.name_scope(None):
      self._should_stop = vs.get_variable(
        f'{self.name}_should_stop',
        shape=[],
        dtype=dtypes.int32,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        initializer=init_ops.constant_initializer(0),
        use_resource=True)
      register_op = self._should_stop.assign(should_stop)
      identity_value_list = []
      with ops.control_dependencies([register_op]):
        for v in value_list:
          if isinstance(v, ops.Tensor):
            identity_value_list.append(array_ops.identity(v))
          elif isinstance(v, sparse_tensor.SparseTensor):
            identity_value_list.append(
              sparse_tensor.SparseTensor(
                array_ops.identity(v.indices),
                array_ops.identity(v.values),
                v.dense_shape))
          else:
            identity_value_list.append(v)
      return nest.pack_sequence_as(values, identity_value_list)

  def sync(self, features=None):
    r'''Calculate should_stop_count.
    '''
    if self._should_stop_count is not None:
      return

    if self._should_stop is None:
      raise ValueError('register should be done before sync')

    with ops.device(self._world[0]), ops.name_scope(None):
      should_stop_counter = vs.get_variable(
        f'{self.name}_should_stop_counter',
        shape=[],
        dtype=dtypes.int32,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        initializer=init_ops.constant_initializer(0),
        use_resource=True)
    deps = None
    if features is not None:
      deps = []
      for v in nest.flatten(features):
        if isinstance(v, ops.Tensor):
          deps.append(v)
        elif isinstance(v, sparse_tensor.SparseTensor):
          deps.append(v.values)
    with ops.control_dependencies(deps):
      self._should_stop_count = should_stop_counter.assign_add(
        self._should_stop)

  def begin(self):
    r'''Called once before using the session.
    '''
    self.sync()

  def before_run(self, run_context):  # pylint: disable=unused-argument
    r'''Call this before sess run.
    '''
    fetches = {}
    if self._should_stop_count is not None:
      fetches['should_stop_count'] = self._should_stop_count
    return session_run_hook.SessionRunArgs(fetches)

  def after_run(self, run_context, run_values):
    r'''Call this after sess run to stop the execution.
    '''
    if 'should_stop_count' in run_values.results:
      should_stop_count = run_values.results['should_stop_count']
      num_workers = len(self._world)
      logging.vlog(
        1, f'{should_stop_count} of {num_workers} workers should stop')
      if self._drop_remainder:
        if should_stop_count > 0:
          logging.info('Stop training since at least one worker should stop')
          run_context.request_stop()
          return
      else:
        if should_stop_count == num_workers:
          logging.info('Stop training since all workers should stop')
          run_context.request_stop()
          return

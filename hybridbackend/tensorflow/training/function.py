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

r'''Function to use HybridBackend.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import random as rn
import threading

import numpy as np
from tensorflow._api.v1 import train as train_v1
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import saver
from tensorflow.python.training import training
from tensorflow.python.util import deprecation

try:
  from tensorflow.python.util import module_wrapper
except ImportError:
  module_wrapper = None

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.context import context_scope
from hybridbackend.tensorflow.framework.device import device_function
from hybridbackend.tensorflow.training.optimizer import wraps_optimizer


def configure(*args, prototype=None, **kwargs):
  r'''Creates ConfigProto.
  '''
  ctx = Context.get()
  if not prototype:
    kwargs.setdefault('allow_soft_placement', True)
    prototype = config_pb2.ConfigProto(*args, **kwargs)
  prototype.graph_options.optimizer_options.global_jit_level = (
    config_pb2.OptimizerOptions.OFF)
  prototype.gpu_options.allow_growth = True
  prototype.gpu_options.force_gpu_compatible = True
  chief = pydev.DeviceSpec.from_string(ctx.devices[0])
  del prototype.device_filters[:]
  prototype.device_filters.append(f'/job:{chief.job}/task:{chief.task}')
  prototype.device_filters.append(
    f'/job:{ctx.task_type}/task:{ctx.task_id}')
  prototype.experimental.collective_group_leader = (
    f'/job:{chief.job}/replica:{chief.replica}/task:{chief.task}')
  return prototype


class PatchTensorflowAPI(object):  # pylint: disable=useless-object-inheritance
  r'''Context manager that patches TF APIs.
  '''
  _lock = threading.Lock()
  _stack_depth = 0

  def __enter__(self):
    with PatchTensorflowAPI._lock:
      PatchTensorflowAPI._stack_depth += 1
      if PatchTensorflowAPI._stack_depth <= 1:

        def decorate_add_weight(fn):
          def decorated(layer, name, shape, **kwargs):
            kwargs['getter'] = vs.get_variable
            if isinstance(layer, base.Layer):
              return fn(layer, name, shape, **kwargs)
            with vs.variable_scope(layer._name):  # pylint: disable=protected-access
              return fn(layer, name, shape, **kwargs)
          return decorated
        self._prev_add_weight = base_layer.Layer.add_weight
        base_layer.Layer.add_weight = decorate_add_weight(self._prev_add_weight)
        self._prev_optimizers = {}

        def decorate_saver_init(fn):
          def decorated(cls, *args, **kwargs):
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
          return decorated
        self._prev_saver_init = saver.Saver.__init__
        saver.Saver.__init__ = decorate_saver_init(self._prev_saver_init)

        for k, c in training.__dict__.items():
          if (isinstance(c, type)
              and issubclass(c, training.Optimizer)
              and c not in (
                training.Optimizer,
                training.SyncReplicasOptimizer)):
            self._prev_optimizers[k] = c
            wrapped = wraps_optimizer(c)
            setattr(training, k, wrapped)
            setattr(train_v1, k, wrapped)
      return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    with PatchTensorflowAPI._lock:
      if PatchTensorflowAPI._stack_depth <= 1:
        base_layer.Layer.add_weight = self._prev_add_weight
        saver.Saver.__init__ = self._prev_saver_init

        for c, opt in self._prev_optimizers.items():
          setattr(training, c, opt)
          setattr(train_v1, c, opt)
      PatchTensorflowAPI._stack_depth -= 1


@contextlib.contextmanager
def scope(**kwargs):
  r'''Context manager that decorates for model construction.
  '''
  seed = kwargs.pop('seed', None)
  if seed is not None:
    rn.seed(seed)
    np.random.seed(seed)
    random_seed.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

  with context_scope(**kwargs) as ctx, ops.device(device_function):
    with PatchTensorflowAPI():
      if module_wrapper is not None:
        with deprecation.silence():
          prev_warning_limit = module_wrapper._PER_MODULE_WARNING_LIMIT  # pylint: disable=protected-access
          module_wrapper._PER_MODULE_WARNING_LIMIT = 0  # pylint: disable=protected-access
          yield ctx
          module_wrapper._PER_MODULE_WARNING_LIMIT = prev_warning_limit  # pylint: disable=protected-access
      else:
        yield ctx


def function(**params):
  r'''Decorator to set params in a function.
  '''
  def decorated(fn):
    def wrapped_fn(*args, **kwargs):
      r'''Wrapped function.
      '''
      with scope(**params):
        return fn(*args, **kwargs)
    return wrapped_fn
  return decorated

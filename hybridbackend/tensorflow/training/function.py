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

import abc
import contextlib
import os
import random as rn
import threading

import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope as vs

from hybridbackend.tensorflow.framework.context import context_scope
from hybridbackend.tensorflow.framework.device import device_function


class Patching(object):  # pylint: disable=useless-object-inheritance
  r'''Python API patching.
  '''
  _lock = threading.Lock()
  _stack_depth = 0
  _registry = {}

  @classmethod
  def register(cls, patching):
    r'''Register implementation.

    Args:
      patching: Implementation class to register.
    '''
    if not issubclass(patching, cls):
      raise ValueError(f'{patching} must be a subclass of Patching')
    cls._registry[patching.__name__] = patching()

  @classmethod
  @contextlib.contextmanager
  def scope(cls):
    r'''Context manager that patches Python APIs.
    '''
    try:
      with cls._lock:
        cls._stack_depth += 1
        if cls._stack_depth <= 1:
          for patching in cls._registry.values():
            patching.patch()
      yield cls
    finally:
      with cls._lock:
        if cls._stack_depth <= 1:
          for patching in cls._registry.values():
            patching.unpatch()
          cls._stack_depth -= 1

  @abc.abstractmethod
  def patch(self):
    r'''Patches APIs.
    '''

  @abc.abstractmethod
  def unpatch(self):
    r'''Revert API patching.
    '''


class PatchingLayers(Patching):
  r'''Patching layers.
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

  def patch(self):
    r'''Patches APIs.
    '''
    self._prev_add_weight = base_layer.Layer.add_weight
    base_layer.Layer.add_weight = self.wraps_add_weight(self._prev_add_weight)

  def unpatch(self):
    r'''Revert API patching.
    '''
    base_layer.Layer.add_weight = self._prev_add_weight


Patching.register(PatchingLayers)


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
    with Patching.scope():
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

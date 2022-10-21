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

r'''Rewriting graph and session run.
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
from tensorflow.python.training import session_run_hook

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.device import device_function
from hybridbackend.tensorflow.framework.ops import ModeKeys


class GraphRewriting(object):  # pylint: disable=useless-object-inheritance
  r'''Python API rewriting.
  '''
  _lock = threading.Lock()
  _stack_depth = 0
  _registry = {}
  _registry_keys = []

  @classmethod
  def register(cls, rewriting):
    r'''Register implementation.

    Args:
      rewriting: Implementation class to register.
    '''
    if not issubclass(rewriting, cls):
      raise ValueError(f'{rewriting} must be a subclass of GraphRewriting')
    if rewriting.__name__ not in cls._registry:
      cls._registry_keys.append(rewriting.__name__)
    cls._registry[rewriting.__name__] = rewriting()

  @classmethod
  @contextlib.contextmanager
  def scope(cls):
    r'''Context manager that patches Python APIs.
    '''
    try:
      at_top = False
      with cls._lock:
        cls._stack_depth += 1
        if cls._stack_depth <= 1:
          for name in cls._registry_keys:
            cls._registry[name].begin()
          at_top = True
      if at_top:
        with ops.device(device_function):
          yield cls
      else:
        yield cls
    finally:
      with cls._lock:
        if cls._stack_depth <= 1:
          for name in reversed(cls._registry_keys):
            cls._registry[name].end()
        cls._stack_depth -= 1

  @abc.abstractmethod
  def begin(self):
    r'''Rewrites API.
    '''

  @abc.abstractmethod
  def end(self):
    r'''Revert API rewriting.
    '''


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

  with Context.scope(**kwargs) as ctx:
    with GraphRewriting.scope():
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


class SessionRunRewriting(session_run_hook.SessionRunHook):
  r'''Collection based SessionRunHook as singleton.
  '''
  _registry = {}
  _mode_keys = {
    ModeKeys.TRAIN: [],
    ModeKeys.EVAL: [],
    ModeKeys.PREDICT: [],
    None: []}

  @classmethod
  def register(cls, hook, modes=None):
    r'''Register implementation.

    Args:
      hook: Implementation class to register.
    '''
    if not issubclass(hook, cls):
      raise ValueError(
        f'{hook} must be a subclass of SessionRunRewriting')
    if hook.__name__ in cls._registry:
      return
    if modes is None:
      modes = [None]
    else:
      modes += [None]
    for mode in set(modes):
      cls._mode_keys[mode].append(hook.__name__)
    cls._registry[hook.__name__] = hook()

  @classmethod
  def hooks(cls):
    r'''Get all registered hooks.
    '''
    return [
      cls._registry[name]
      for name in cls._mode_keys[Context.get().options.mode]]

  @classmethod
  def collection_name(cls, name):
    r'''Get name of related collection.
    '''
    if Context.get().options.mode is not None:
      return f'{Context.get().options.mode}_{name}'
    return name

  @classmethod
  def add_to_collection(cls, name, op):
    r'''Add op to related collection.
    '''
    return ops.add_to_collection(SessionRunRewriting.collection_name(name), op)

  @classmethod
  def get_collection(cls, name):
    r'''Get related collection.
    '''
    return ops.get_collection(SessionRunRewriting.collection_name(name))

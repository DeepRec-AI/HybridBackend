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

from tensorflow.python.framework import ops
from tensorflow.python.keras.backend import reset_uids as reset_keras_uids
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variable_scope as vs

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.scope import scope


class VariableBlock(object):  # pylint: disable=useless-object-inheritance
  r'''Function scope that manages variables.
  '''
  def __init__(self):
    self._reuse_variables = Context.get().param('reuse_variables', True)

  def begin(self):
    self._prev = vs.get_variable_scope()._reuse  # pylint: disable=protected-access

  def reset(self):
    reset_keras_uids()
    varscope = ops.get_default_graph().get_collection_ref(('__varscope',))
    if varscope:
      varscope[0].variable_scopes_count.clear()
    if self._reuse_variables:
      vs.get_variable_scope().reuse_variables()

  def end(self):
    if self._reuse_variables:
      vs.get_variable_scope()._reuse = self._prev  # pylint: disable=protected-access

  def __enter__(self):
    self.begin()
    self.reset()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.end()


@contextlib.contextmanager
def function_extent():
  r'''Context manager that decorates TF APIs.
  '''
  try:
    def decorated_metric_variable(
        shape, dtype, validate_shape=True, name=None):
      return vs.get_variable(
        name,
        initializer=array_ops.zeros(shape, dtype),
        trainable=False,
        collections=[
          ops.GraphKeys.LOCAL_VARIABLES,
          ops.GraphKeys.METRIC_VARIABLES],
        validate_shape=validate_shape,
        synchronization=vs.VariableSynchronization.ON_READ,
        aggregation=vs.VariableAggregation.SUM)
    prev_metric_variable = metrics_impl.metric_variable
    metrics_impl.metric_variable = decorated_metric_variable

    def decorate_add_weight(fn):
      def decorated(layer, name, shape, **kwargs):
        kwargs['getter'] = vs.get_variable
        return fn(layer, name, shape, **kwargs)
      return decorated
    prev_add_weight = base_layer.Layer.add_weight
    base_layer.Layer.add_weight = decorate_add_weight(prev_add_weight)

    yield
  finally:
    metrics_impl.metric_variable = prev_metric_variable
    base_layer.Layer.add_weight = prev_add_weight


def function(**params):
  r'''Decorator to set params in a function.
  '''
  def decorated(fn):
    def wrapped_fn(*args, **kwargs):
      r'''Wrapped function.
      '''
      with scope(**params):
        with function_extent():
          return fn(*args, **kwargs)
    return wrapped_fn
  return decorated

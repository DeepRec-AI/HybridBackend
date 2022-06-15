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

r'''Variable utility for training.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.framework import ops
from tensorflow.python.keras.backend import reset_uids as reset_keras_uids
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs


class ReuseVariables(object):  # pylint: disable=useless-object-inheritance
  r'''Variable reusing context.
  '''
  def __call__(self, reuse):
    reset_keras_uids()
    varscope = ops.get_default_graph().get_collection_ref(('__varscope',))
    if varscope:
      varscope[0].variable_scopes_count.clear()
    vs.get_variable_scope()._reuse = reuse  # pylint: disable=protected-access


@contextlib.contextmanager
def reuse_variables(reuse=None):
  r'''Context manager that reuses variables.
  '''
  try:
    fn = ReuseVariables()
    prev_reuse = vs.get_variable_scope()._reuse  # pylint: disable=protected-access
    if reuse is not None:
      fn(reuse)
    yield fn
  finally:
    vs.get_variable_scope()._reuse = prev_reuse  # pylint: disable=protected-access


@contextlib.contextmanager
def disable_variable_update():
  r'''Context manager that disable update in state_ops's assign operations
  '''
  try:
    def wraps_assign(assign_fn):  # pylint: disable=unused-argument
      r'''Disable the assign op
      '''
      def wrapped_assign(
          ref, value, validate_shape=None, use_locking=None, name=None):  # pylint: disable=unused-argument
        return value
      return wrapped_assign

    def wraps_assign_sub(assign_sub_fn):  # pylint: disable=unused-argument
      r'''Disable the assign_sub op
      '''
      def wrapped_assign_sub(ref, value, use_locking=None, name=None):  # pylint: disable=unused-argument
        return math_ops.subtract(ref, value)
      return wrapped_assign_sub

    def wraps_assign_add(assign_add_fn):  # pylint: disable=unused-argument
      r'''Disable the assign_add op
      '''
      def wrapped_assign_add(ref, value, use_locking=None, name=None):  # pylint: disable=unused-argument
        return math_ops.add(ref, value)
      return wrapped_assign_add

    prev_assign = state_ops.assign
    state_ops.assign = wraps_assign(prev_assign)
    prev_assign_sub = state_ops.assign_sub
    state_ops.assign_sub = wraps_assign_sub(prev_assign_sub)
    prev_assign_add = state_ops.assign_add
    state_ops.assign_add = wraps_assign_add(prev_assign_add)

    yield

  finally:
    state_ops.assign = prev_assign
    state_ops.assign_sub = prev_assign_sub
    state_ops.assign_add = prev_assign_add

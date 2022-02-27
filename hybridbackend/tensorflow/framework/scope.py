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

r'''Context manager to use HybridBackend globally.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.framework import ops
from tensorflow.python.keras.backend import reset_uids as reset_keras_uids

from hybridbackend.tensorflow.framework.context import Context


@contextlib.contextmanager
def scope(**kwargs):
  r'''Update params in conext.
  '''
  try:
    ctx = Context.get()
    prev_kwargs = {k: ctx.get_param(k) for k in kwargs if ctx.has_param(k)}
    ctx.update_params(**kwargs)
    yield ctx
  finally:
    ctx.update_params(**prev_kwargs)
    del prev_kwargs


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


def reset_keras_var_count():
  r'''Reset variable scope counts of keras.
  '''
  reset_keras_uids()
  varscope = ops.get_default_graph().get_collection_ref(('__varscope',))
  if varscope:
    varscope[0].variable_scopes_count.clear()

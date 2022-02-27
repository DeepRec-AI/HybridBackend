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
import numpy as np
import os
import random as rn

from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed

from hybridbackend.tensorflow.framework.device import device_function


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

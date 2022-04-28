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

r'''Decorator to wraps customized object.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import estimator
from tensorflow.python.training import optimizer

from hybridbackend.tensorflow.estimator.estimator import wraps_estimator
from hybridbackend.tensorflow.training.optimizer import wraps_optimizer


def wraps(cls):
  r'''Wraps object to be used in HybridBackend.
  '''
  if issubclass(cls, optimizer.Optimizer):
    return wraps_optimizer(cls)
  if issubclass(cls, estimator.Estimator):
    return wraps_estimator(cls)
  return cls

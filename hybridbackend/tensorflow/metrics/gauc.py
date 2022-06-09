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

r'''A data-parallel gAUC metric.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

from hybridbackend.tensorflow.common import oplib as _ops
from hybridbackend.tensorflow.metrics.mean import mean


def gauc(labels,
         predictions,
         indicators=None,
         metrics_collections=None,
         updates_collections=None,
         name=None):
  r'''Computes the approximate gAUC.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    indicators: A `Tensor` whose shape matches `predictions`.
    metrics_collections: An optional list of collections that `mean`
      should be added to.
    updates_collections: An optional list of collections that `update_op`
      should be added to.
    name: An optional variable_scope name.

  Returns:
    (gauc, update_op): A tuple of a scalar `Tensor` representing the current
      g-area-under-curve and an operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `auc`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  '''
  if indicators is None:
    indicators = math_ops.range(
      0, array_ops.shape(array_ops.reshape(labels, [-1]))[0],
      dtype=dtypes.int32)
  with vs.variable_scope(name, 'gauc', (labels, predictions, indicators)):
    aucs, counts = _ops.hb_gauc_calc(labels, predictions, indicators)
    return mean(aucs, counts, metrics_collections, updates_collections, name)

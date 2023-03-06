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

r'''A data-parallel Mean metric.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import weights_broadcast_ops

from hybridbackend.tensorflow.distribute.collective import Collective
from hybridbackend.tensorflow.distribute.ops import CollectiveOps


def mean(values,
         weights=None,
         metrics_collections=None,
         updates_collections=None,
         name=None):
  r'''Computes the (weighted) mean of the given values.

  The `mean` function creates two local variables, `total` and `count`
  that are used to compute the average of `values`. This average is ultimately
  returned as `mean` which is an idempotent operation that simply divides
  `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean`.
  `update_op` increments `total` with the reduced sum of the product of `values`
  and `weights`, and it increments `count` with the reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `Tensor` of arbitrary dimensions.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `values`, and must be broadcastable to `values` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `values` dimension).
    metrics_collections: An optional list of collections that `mean`
      should be added to.
    updates_collections: An optional list of collections that `update_op`
      should be added to.
    name: An optional variable_scope name.

  Returns:
    mean: A `Tensor` representing the current mean, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_value`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
    RuntimeError: If eager execution is enabled.
  '''
  with vs.variable_scope(name, 'mean', (values, weights)):
    values = math_ops.to_float(values)

    total = metrics_impl.metric_variable([], dtypes.float32, name='total')
    count = metrics_impl.metric_variable([], dtypes.float32, name='count')

    if weights is None:
      num_values = math_ops.to_float(array_ops.size(values))
      values_sum = math_ops.reduce_sum(values)
    else:
      values, _, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
        predictions=values, labels=None, weights=weights)
      weights = weights_broadcast_ops.broadcast_weights(
        math_ops.to_float(weights), values)
      values = math_ops.multiply(values, weights)
      values_sum = math_ops.reduce_sum(values)
      num_values = math_ops.reduce_sum(weights)

    stacked = array_ops.stack([values_sum, num_values])
    if isinstance(stacked, (list, tuple)):
      stacked = stacked[0]
    sum_stacked = Collective.get().allreduce(
      stacked, reduce_op=CollectiveOps.SUM)
    if isinstance(sum_stacked, (list, tuple)):
      sum_stacked = sum_stacked[0]
    values_sum, num_values = array_ops.unstack(sum_stacked)

    update_total_op = state_ops.assign_add(total, values_sum)
    with ops.control_dependencies([values]):
      update_count_op = state_ops.assign_add(count, num_values)
    # pylint: disable=protected-access
    metric_op = (
      metrics_impl._safe_scalar_div(total, count, 'value')
      if hasattr(metrics_impl, '_safe_scalar_div')
      else metrics_impl._safe_div(total, count, 'value'))

    if metrics_collections:
      ops.add_to_collections(metrics_collections, metric_op)

    # pylint: disable=protected-access
    update_op = (
      metrics_impl._safe_scalar_div(
        update_total_op, update_count_op, 'update_op')
      if hasattr(metrics_impl, '_safe_scalar_div')
      else metrics_impl._safe_div(
        update_total_op, update_count_op, 'update_op'))

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return metric_op, update_op

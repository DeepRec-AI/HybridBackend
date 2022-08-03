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

r'''A data-parallel AUC metric.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging

from hybridbackend.tensorflow.distribute.communicator import CollectiveOps
from hybridbackend.tensorflow.distribute.communicator_pool import \
  CommunicatorPool


def _allreduce_auc(comm, inputs, inputs_deps):
  r'''Communicator call to reduce auc across workers.
  '''
  with ops.control_dependencies(inputs_deps):
    if isinstance(inputs, (list, tuple)):
      inputs = inputs[0]
    sum_inputs = comm.allreduce(inputs, CollectiveOps.SUM)
    return sum_inputs, None


def _confusion_matrix_at_thresholds(labels,
                                    predictions,
                                    thresholds,
                                    weights=None):
  r'''Computes true_positives, false_negatives, true_negatives, false_positives.

  This function creates up to four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives`.
  `true_positive[i]` is defined as the total weight of values in `predictions`
  above `thresholds[i]` whose corresponding entry in `labels` is `True`.
  `false_negatives[i]` is defined as the total weight of values in `predictions`
  at most `thresholds[i]` whose corresponding entry in `labels` is `True`.
  `true_negatives[i]` is defined as the total weight of values in `predictions`
  at most `thresholds[i]` whose corresponding entry in `labels` is `False`.
  `false_positives[i]` is defined as the total weight of values in `predictions`
  above `thresholds[i]` whose corresponding entry in `labels` is `False`.

  For estimation of these metrics over a stream of data, for each metric the
  function respectively creates an `update_op` operation that updates the
  variable and returns its value.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).

  Returns:
    values: Dict of variables of shape `[len(thresholds)]`.
    update_ops: Dict of operations that increments the `values`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`.
  '''
  with ops.control_dependencies([
      check_ops.assert_greater_equal(
        predictions,
        math_ops.cast(0.0, dtype=predictions.dtype),
        message='predictions must be in [0, 1]'),
      check_ops.assert_less_equal(
        predictions,
        math_ops.cast(1.0, dtype=predictions.dtype),
        message='predictions must be in [0, 1]')]):
    predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
      predictions=math_ops.to_float(predictions),
      labels=math_ops.cast(labels, dtype=dtypes.bool),
      weights=weights)

  num_thresholds = len(thresholds)

  # Reshape predictions and labels.
  predictions_2d = array_ops.reshape(predictions, [-1, 1])
  labels_2d = array_ops.reshape(
    math_ops.cast(labels, dtype=dtypes.bool), [1, -1])

  # Use static shape if known.
  num_predictions = predictions_2d.get_shape().as_list()[0]

  # Otherwise use dynamic shape.
  if num_predictions is None:
    num_predictions = array_ops.shape(predictions_2d)[0]
  thresh_tiled = array_ops.tile(
    array_ops.expand_dims(array_ops.constant(thresholds), [1]),
    array_ops.stack([1, num_predictions]))

  # Tile the predictions after thresholding them across different thresholds.
  pred_is_pos = math_ops.greater(
    array_ops.tile(array_ops.transpose(predictions_2d), [num_thresholds, 1]),
    thresh_tiled)
  pred_is_neg = math_ops.logical_not(pred_is_pos)

  # Tile labels by number of thresholds
  label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])
  label_is_neg = math_ops.logical_not(label_is_pos)

  if weights is not None:
    weights = weights_broadcast_ops.broadcast_weights(
      math_ops.to_float(weights), predictions)
    weights_tiled = array_ops.tile(
      array_ops.reshape(weights, [1, -1]), [num_thresholds, 1])
    thresh_tiled.get_shape().assert_is_compatible_with(
      weights_tiled.get_shape())
  else:
    weights_tiled = None

  values = {}
  update_ops = {}

  true_p = metrics_impl.metric_variable(
    [num_thresholds], dtypes.float32, name='true_positives')
  is_true_positive = math_ops.to_float(
    math_ops.logical_and(label_is_pos, pred_is_pos))
  if weights_tiled is not None:
    is_true_positive *= weights_tiled

  false_n = metrics_impl.metric_variable(
    [num_thresholds], dtypes.float32, name='false_negatives')
  is_false_negative = math_ops.to_float(
    math_ops.logical_and(label_is_pos, pred_is_neg))
  if weights_tiled is not None:
    is_false_negative *= weights_tiled

  true_n = metrics_impl.metric_variable(
    [num_thresholds], dtypes.float32, name='true_negatives')
  is_true_negative = math_ops.to_float(
    math_ops.logical_and(label_is_neg, pred_is_neg))
  if weights_tiled is not None:
    is_true_negative *= weights_tiled

  false_p = metrics_impl.metric_variable(
    [num_thresholds], dtypes.float32, name='false_positives')
  is_false_positive = math_ops.to_float(
    math_ops.logical_and(label_is_neg, pred_is_pos))
  if weights_tiled is not None:
    is_false_positive *= weights_tiled

  tp_sum = math_ops.reduce_sum(is_true_positive, 1)
  fn_sum = math_ops.reduce_sum(is_false_negative, 1)
  tn_sum = math_ops.reduce_sum(is_true_negative, 1)
  fp_sum = math_ops.reduce_sum(is_false_positive, 1)

  stacked = array_ops.stack([tp_sum, fn_sum, tn_sum, fp_sum])
  sum_stacked = CommunicatorPool.get().call(
    _allreduce_auc, stacked, trainable=False)
  if isinstance(sum_stacked, (list, tuple)):
    sum_stacked = sum_stacked[0]
  tp_sum, fn_sum, tn_sum, fp_sum = array_ops.unstack(sum_stacked)

  update_ops['tp'] = state_ops.assign_add(true_p, tp_sum)
  update_ops['fn'] = state_ops.assign_add(false_n, fn_sum)
  update_ops['tn'] = state_ops.assign_add(true_n, tn_sum)
  update_ops['fp'] = state_ops.assign_add(false_p, fp_sum)

  values['tp'] = true_p
  values['fn'] = false_n
  values['tn'] = true_n
  values['fp'] = false_p

  return values, update_ops


def auc(labels,
        predictions,
        weights=None,
        num_thresholds=200,
        metrics_collections=None,
        updates_collections=None,
        curve='ROC',
        name=None,
        summation_method='trapezoidal'):
  r'''Computes the approximate AUC via a Riemann sum.

  The `auc` function creates four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives` that are used to
  compute the AUC. To discretize the AUC curve, a linearly spaced set of
  thresholds is used to compute pairs of recall and precision values. The area
  under the ROC-curve is therefore computed using the height of the recall
  values by the false positive rate, while the area under the PR-curve is the
  computed using the height of the precision values by the recall.

  This value is ultimately returned as `auc`, an idempotent operation that
  computes the area under a discretized curve of precision versus recall values
  (computed using the aforementioned variables). The `num_thresholds` variable
  controls the degree of discretization with larger numbers of thresholds more
  closely approximating the true AUC. The quality of the approximation may vary
  dramatically depending on `num_thresholds`.

  For best results, `predictions` should be distributed approximately uniformly
  in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC
  approximation may be poor if this is not the case. Setting `summation_method`
  to 'minoring' or 'majoring' can help quantify the error in the approximation
  by providing lower or upper bound estimate of the AUC.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `auc`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use when discretizing the roc
      curve.
    metrics_collections: An optional list of collections that `auc` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    curve: Specifies the name of the curve to be computed, 'ROC' [default] or
      'PR' for the Precision-Recall-curve.
    name: An optional variable_scope name.
    summation_method: Specifies the Riemann summation method used
      (https://en.wikipedia.org/wiki/Riemann_sum): 'trapezoidal' [default] that
      applies the trapezoidal rule; 'careful_interpolation', a variant of it
      differing only by a more correct interpolation scheme for PR-AUC -
      interpolating (true/false) positives but not the ratio that is precision;
      'minoring' that applies left summation for increasing intervals and right
      summation for decreasing intervals; 'majoring' that does the opposite.
      Note that 'careful_interpolation' is strictly preferred to 'trapezoidal'
      (to be deprecated soon) as it applies the same method for ROC, and a
      better one (see Davis & Goadrich 2006 for details) for the PR curve.

  Returns:
    (auc, update_op): A tuple of a scalar `Tensor` representing the current
      area-under-curve and an operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `auc`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  '''
  with vs.variable_scope(name, 'auc', (labels, predictions, weights)):
    if curve not in ('ROC', 'PR'):
      raise ValueError(f'curve must be either ROC or PR, {curve} unknown')
    kepsilon = 1e-7  # to account for floating point imprecisions
    thresholds = [
      (i + 1) * 1.0 / (num_thresholds - 1)
      for i in range(num_thresholds - 2)]
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

    values, update_ops = _confusion_matrix_at_thresholds(
      labels, predictions, thresholds, weights)

    # Add epsilons to avoid dividing by 0.
    epsilon = 1.0e-6

    def interpolate_pr_auc(tp, fp, fn):
      r'''Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

      Note here we derive & use a closed formula not present in the paper
      - as follows:
      Modeling all of TP (true positive weight),
      FP (false positive weight) and their sum P = TP + FP (positive weight)
      as varying linearly within each interval [A, B] between successive
      thresholds, we get
        Precision = (TP_A + slope * (P - P_A)) / P
      with slope = dTP / dP = (TP_B - TP_A) / (P_B - P_A).
      The area within the interval is thus (slope / total_pos_weight) times
        int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
        int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}
      where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in
        int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)
      Bringing back the factor (slope / total_pos_weight) we'd put aside, we get
         slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight
      where dTP == TP_B - TP_A.
      Note that when P_A == 0 the above calculation simplifies into
        int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)
      which is really equivalent to imputing constant precision throughout the
      first bucket having >0 true positives.

      Args:
        tp: true positive counts
        fp: false positive counts
        fn: false negative counts
      Returns:
        pr_auc: an approximation of the area under the P-R curve.
      '''
      dtp = tp[:num_thresholds - 1] - tp[1:]
      p = tp + fp
      prec_slope = metrics_impl._safe_div(  # pylint: disable=protected-access
        dtp, p[:num_thresholds - 1] - p[1:], 'prec_slope')
      intercept = tp[1:] - math_ops.multiply(prec_slope, p[1:])
      safe_p_ratio = array_ops.where(
        math_ops.logical_and(p[:num_thresholds - 1] > 0, p[1:] > 0),
        metrics_impl._safe_div(  # pylint: disable=protected-access
          p[:num_thresholds - 1], p[1:], 'recall_relative_ratio'),
        array_ops.ones_like(p[1:]))
      return math_ops.reduce_sum(
        metrics_impl._safe_div(  # pylint: disable=protected-access
          prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)),
          tp[1:] + fn[1:],
          name='pr_auc_increment'),
        name='interpolate_pr_auc')

    def compute_auc(tp, fn, tn, fp, name):
      r'''Computes the roc-auc or pr-auc based on confusion counts.
      '''
      if curve == 'PR':
        if summation_method == 'trapezoidal':
          logging.warning(
            'Trapezoidal rule is known to produce incorrect PR-AUCs; '
            'please switch to "careful_interpolation" instead.')
        elif summation_method == 'careful_interpolation':
          # This one is a bit tricky and is handled separately.
          return interpolate_pr_auc(tp, fp, fn)
      rec = math_ops.div(tp + epsilon, tp + fn + epsilon)
      if curve == 'ROC':
        fp_rate = math_ops.div(fp, fp + tn + epsilon)
        x = fp_rate
        y = rec
      else:  # curve == 'PR'.
        prec = math_ops.div(tp + epsilon, tp + fp + epsilon)
        x = rec
        y = prec
      if summation_method in ('trapezoidal', 'careful_interpolation'):
        # Note that the case ('PR', 'careful_interpolation') has been handled
        # above.
        return math_ops.reduce_sum(
          math_ops.multiply(
            x[:num_thresholds - 1] - x[1:],
            (y[:num_thresholds - 1] + y[1:]) / 2.),
          name=name)
      if summation_method == 'minoring':
        return math_ops.reduce_sum(
          math_ops.multiply(
            x[:num_thresholds - 1] - x[1:],
            math_ops.minimum(y[:num_thresholds - 1], y[1:])),
          name=name)
      if summation_method == 'majoring':
        return math_ops.reduce_sum(
          math_ops.multiply(
            x[:num_thresholds - 1] - x[1:],
            math_ops.maximum(y[:num_thresholds - 1], y[1:])),
          name=name)
      raise ValueError(f'Invalid summation_method: {summation_method}')

    auc_value = compute_auc(
      values['tp'], values['fn'], values['tn'], values['fp'], 'value')
    if metrics_collections:
      ops.add_to_collections(metrics_collections, auc_value)

    update_op = compute_auc(
      update_ops['tp'],
      update_ops['fn'],
      update_ops['tn'],
      update_ops['fp'],
      'update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return auc_value, update_op

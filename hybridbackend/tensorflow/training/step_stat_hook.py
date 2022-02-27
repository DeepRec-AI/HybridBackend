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

r'''Session run hook for benchmarking.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

from tensorflow.python.framework import ops
from tensorflow.python.training import session_run_hook
from tensorflow.python.platform import tf_logging as logging


class StepStatHook(session_run_hook.SessionRunHook):
  r'''Hook that counts performance statistics for steps.
  '''
  def __init__(self, every_n_iter=None, count=None, unit='sample'):
    self._every_n_iter = every_n_iter
    self._count = count
    self._unit = unit
    self._count_runnable = isinstance(self._count, (ops.Operation, ops.Tensor))

  @property
  def should_print_logs(self):
    return (
      self._every_n_iter is not None
      and self._iter_count > 0
      and self._iter_count % self._every_n_iter == 0)

  def begin(self):
    r'''Called once before using the session.
    '''
    self._iter_count = 0
    self._durations = []
    self._counts = []
    self._prev_ts = None

  def before_run(self, run_context):
    r'''Called before each call to run().
    '''
    _ = run_context
    self._prev_ts = time.time()
    if self._count_runnable:
      return session_run_hook.SessionRunArgs(self._count)
    return None

  def after_run(self, run_context, run_values):
    r'''Called after each call to run().
    '''
    self._durations.append(time.time() - self._prev_ts)
    _ = run_context
    if self._count is not None:
      if self._count_runnable:
        self._counts.append(run_values.results)
      else:
        self._counts.append(self._count)
      if self.should_print_logs:
        durs = np.array(self._durations)
        cnts = np.array(self._counts)
        p50 = np.percentile(durs, 50)
        p10 = np.percentile(durs, 10)
        p90 = np.percentile(durs, 90)
        fps = 1. * np.sum(cnts) / np.sum(durs)
        logging.info(
          f'secs/step: {p50:.5f} ({100.*p10/p90:.2f}%), '
          f'{self._unit}s/sec: {fps:.2f}')
        self._durations = []
        self._counts = []
    else:
      if self.should_print_logs:
        durs = np.array(self._durations)
        p50 = np.percentile(durs, 50)
        p10 = np.percentile(durs, 10)
        p90 = np.percentile(durs, 90)
        logging.info(f'secs/step: {p50:.5f} ({100.*p10/p90:.2f}%)')
        self._durations = []
    self._iter_count += 1

  def end(self, session):
    r'''Called at the end of session.
    '''
    _ = session
    if self._count is not None:
      durs = np.array(self._durations)
      cnts = np.array(self._counts)
      if self._durations:
        p50 = np.percentile(durs, 50)
        p10 = np.percentile(durs, 10)
        p90 = np.percentile(durs, 90)
        fps = 1. * np.sum(cnts) / np.sum(durs)
        logging.info(
          f'secs/step: {p50:.5f} ({100.*p10/p90:.2f}%), '
          f'{self._unit}s/sec: {fps:.2f}')
      self._durations = []
      self._counts = []
    else:
      durs = np.array(self._durations)
      if self._durations:
        p50 = np.percentile(durs, 50)
        p10 = np.percentile(durs, 10)
        p90 = np.percentile(durs, 90)
        logging.info(f'secs/step: {p50:.5f} ({100.*p10/p90:.2f}%)')
      self._durations = []

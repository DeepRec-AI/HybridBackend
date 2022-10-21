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

r'''Iterator to iterate tensors by prefetching.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from six.moves import xrange
import threading
import weakref

from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from hybridbackend.tensorflow.common import oplib as _ops
from hybridbackend.tensorflow.framework.ops import ModeKeys
from hybridbackend.tensorflow.framework.rewriting import SessionRunRewriting

ops.NotDifferentiable('HbPrefetchBufferPut')
ops.NotDifferentiable('HbPrefetchBufferTake')
ops.NotDifferentiable('HbPrefetchBufferCancel')
ops.NotDifferentiable('HbPrefetchBufferClose')
ops.NotDifferentiable('HbPrefetchBufferSize')


class Iterator(object):  # pylint: disable=useless-object-inheritance
  r'''Iterate tensors with prefetching.

  The `Iterator`, combined with the `Coordinator` provides a way to
  compute tensors asynchronously using multiple threads.
  '''
  def __init__(
      self,
      inputs,
      capacity=1,
      num_runners=1,
      num_takers=1,
      closed_exception_types=None,
      ignored_exception_types=None):
    r'''Create a Iterator.

    When you later call the `create_threads()` method, the `Iterator` will
    create threads for prefetching inputs. Each thread will run in parallel.

    Args:
      inputs: Dataset iterator or nest structure of tensors.
      capacity: (Optional.) Max number of records to keep in the buffer.
      num_runners: (Optional.) Number of threads for prefetching. 1 by
        default.
      num_takers: (Optional.) Number of threads for taking next tensors. 1 by
        default.
      closed_exception_types: (Optional.) Exception types indicating that the
        prefetching is normally finished. Defaults to
        `(tf.errors.OutOfRangeError, StopIteration)`.
      ignored_exception_types: (Optional.) Exception types indicating that the
        prefetching can continue. Defaults to `()`.
    '''
    try:
      executing_eagerly = context.executing_eagerly()
    except:  # pylint: disable=bare-except
      executing_eagerly = context.in_eager_mode()
    else:
      executing_eagerly = False
    if not executing_eagerly:
      self._name = ops.get_default_graph().unique_name(self.__class__.__name__)
    else:
      self._name = context.context().scope_name
    if not closed_exception_types:
      self._closed_exception_types = (errors.OutOfRangeError, StopIteration)
    else:
      self._closed_exception_types = tuple(closed_exception_types)
    if not ignored_exception_types:
      self._ignored_exception_types = ()
    else:
      self._ignored_exception_types = tuple(ignored_exception_types)
    self._lock = threading.Lock()
    self._runs_per_session = weakref.WeakKeyDictionary()
    self._exceptions_raised = []

    with ops.name_scope(self._name):
      self._cancel_op = _ops.hb_prefetch_buffer_cancel(
        shared_name=self._name,
        shared_capacity=capacity)
      self._resume_op = _ops.hb_prefetch_buffer_cancel(
        is_cancelled=False,
        shared_name=self._name,
        shared_capacity=capacity)
      self._close_op = _ops.hb_prefetch_buffer_close(
        shared_name=self._name,
        shared_capacity=capacity)
      self._runner_per_thread = []

      if isinstance(inputs, iterator_ops.Iterator):
        inputs = inputs.get_next()
      tensor_or_sparse_tensor_or_nones = nest.flatten(inputs)
      tensor_or_nones = []
      for t in tensor_or_sparse_tensor_or_nones:
        if hasattr(t, 'dense_shape'):
          tensor_or_nones.extend([t.values, t.indices, t.dense_shape])
        else:
          tensor_or_nones.append(t)
      tensor_indices = []
      tensors = []
      for i, v in enumerate(tensor_or_nones):
        if v is not None:
          tensor_indices.append(i)
          tensors.append(v)
      runner = _ops.hb_prefetch_buffer_put(
        tensors,
        shared_name=self._name,
        shared_capacity=capacity)
      tensor_dtypes = []
      tensor_shapes = []
      for v in tensors:
        tensor_dtypes.append(v.dtype)
        tensor_shapes.append(v.shape if hasattr(v, 'shape') else None)
      self._runner_per_thread = [runner for _ in range(num_runners)]

      num_takers = max(num_takers, capacity)
      next_tensors = _ops.hb_prefetch_buffer_take(
        dtypes=tensor_dtypes,
        shared_name=self._name,
        shared_capacity=capacity,
        shared_threads=num_takers)
      if not isinstance(next_tensors, (tuple, list)):
        next_tensors = [next_tensors]
      next_tensors = [array_ops.identity(t) for t in next_tensors]
      for i, t in enumerate(next_tensors):
        t.set_shape(tensor_shapes[i])
      next_tensor_or_nones = [None] * len(tensor_or_nones)
      for i, v in enumerate(next_tensors):
        next_tensor_or_nones[tensor_indices[i]] = v
      next_tensor_or_nones = collections.deque(next_tensor_or_nones)
      next_tensor_or_sparse_tensor_or_nones = []
      for t in tensor_or_sparse_tensor_or_nones:
        if hasattr(t, 'dense_shape'):
          sparse_values = next_tensor_or_nones.popleft()
          sparse_indices = next_tensor_or_nones.popleft()
          sparse_dense_shape = next_tensor_or_nones.popleft()
          next_tensor_or_sparse_tensor_or_nones.append(
            sparse_tensor.SparseTensor(
              values=sparse_values,
              indices=sparse_indices,
              dense_shape=sparse_dense_shape))
        else:
          next_tensor_or_sparse_tensor_or_nones.append(
            next_tensor_or_nones.popleft())
      self._next_inputs = nest.pack_sequence_as(
        inputs, next_tensor_or_sparse_tensor_or_nones)

    self._created_threads = False
    ops.add_to_collection(self.__class__.__name__, self)

  @property
  def name(self):
    r'''Name of this iterator.
    '''
    return self._name

  @property
  def num_runners(self):
    r'''The number of running threads.
    '''
    return len(self._runner_per_thread)

  @property
  def closed_exception_types(self):
    r'''Exception types indicating that prefetching is normally finished.
    '''
    return self._closed_exception_types

  @property
  def exceptions_raised(self):
    r'''Exceptions raised but not handled by the `Iterator` threads.

    Exceptions raised in `Iterator` threads are handled in one of two ways
    depending on whether or not a `Coordinator` was passed to
    `create_threads()`:

    * With a `Coordinator`, exceptions are reported to the coordinator and
      forgotten by the `Iterator`.
    * Without a `Coordinator`, exceptions are captured by the `Iterator`
      and made available in this `exceptions_raised` property.

    Returns:
      A list of Python `Exception` objects.  The list is empty if no exception
      was captured.  (No exceptions are captured when using a Coordinator.)
    '''
    return self._exceptions_raised

  def get_next(self):
    r'''Get next inputs
    '''
    return self._next_inputs

  # pylint: disable=broad-except
  def _run(self, sess, coord, index):
    r'''Run prefetching in thread.

    Args:
      sess: A `Session`.
      coord: A `Coordinator` object for reporting errors and checking stop
        conditions.
      index: Index of current thread.
    '''
    decremented = False
    try:
      sess.run(self._resume_op)
      run_fetch = sess.make_callable(self._runner_per_thread[index])
      while True:
        try:
          # Use `next` instead of `for .. in` to reraise exception in generator.
          if coord and coord.should_stop():
            break
          run_fetch()
        except errors.CancelledError:
          logging.vlog(1, 'Prefetching was cancelled.')
          return
        except self._closed_exception_types:  # pylint: disable=catching-non-exception
          logging.vlog(1, 'Prefetching was closed.')
          with self._lock:
            self._runs_per_session[sess] -= 1
            decremented = True
            if self._runs_per_session[sess] == 0:
              try:
                sess.run(self._close_op)
              except Exception:
                pass
            return
        except self._ignored_exception_types as e:  # pylint: disable=catching-non-exception
          logging.warning(
            'Corrupted inputs were ignored in prefetching:\n\n%s', e)
          continue
    except Exception as e:
      if coord:
        coord.request_stop(e)
        if not isinstance(e, errors.CancelledError) and \
           not isinstance(e, self._closed_exception_types) and \
           not isinstance(e, self._ignored_exception_types):
          logging.error(
            'Prefetching was cancelled unexpectedly:\n\n%s', e)
          raise
      else:
        with self._lock:
          self._exceptions_raised.append(e)
        raise
    finally:
      if not decremented:
        with self._lock:
          self._runs_per_session[sess] -= 1

  def _cancel_on_stop(self, sess, coord):
    r'''Clean up resources on stop.

    Args:
      sess: A `Session`.
      coord: A `Coordinator` object for reporting errors and checking stop
        conditions.
    '''
    coord.wait_for_stop()
    try:
      sess.run(self._cancel_op)
    except Exception:
      pass
  # pylint: enable=broad-except

  def create_threads(self, sess, coord=None, daemon=True, start=True):
    r'''Create threads to prefetch for the given session.

    This method requires a session in which the graph was launched. It creates
    a list of threads, optionally starting them.

    The `coord` argument is an optional coordinator that the threads will use
    to terminate together and report exceptions.  If a coordinator is given,
    this method starts an additional thread to cancel when the coordinator
    requests a stop.

    If previously created threads for the given session are still running, no
    new threads will be created.

    Args:
      sess: A `Session`.
      coord: (Optional.) `Coordinator` object for reporting errors and checking
        stop conditions.
      daemon: (Optional.) Boolean. If `True` make the threads daemon threads.
      start: (Optional.) Boolean. If `True` starts the threads.  If `False` the
        caller must call the `start()` method of the returned threads.

    Returns:
      A list of threads.
    '''
    if self._created_threads:
      return []

    with self._lock:
      try:
        if self._runs_per_session[sess] > 0:
          # Already started: no new threads to return.
          return []
      except KeyError:
        pass
      self._runs_per_session[sess] = self.num_runners
      self._exceptions_raised = []

    ret_threads = []
    for i in xrange(self.num_runners):
      ret_threads.append(threading.Thread(
        target=self._run,
        args=(sess, coord, i),
        name=f'PrefetchThread-{self.name}-{i}'))
    if coord:
      name = f'CancelOnStopThread-{self.name}'
      ret_threads.append(threading.Thread(
        target=self._cancel_on_stop,
        args=(sess, coord),
        name=name))
    for t in ret_threads:
      if coord:
        coord.register_thread(t)
      if daemon:
        t.daemon = True
      if start:
        t.start()
    self._created_threads = True
    return ret_threads

  @classmethod
  def start(cls, sess, coord=None):
    r'''Start threads to prefetch tensors for the given session.

    Args:
      sess: A `Session`.
      coord: (Optional.) `Coordinator` object for reporting errors and checking
        stop conditions.

    Returns:
      A list of threads.
    '''
    if sess is None:
      sess = ops.get_default_session()
      if not sess:
        raise ValueError(
          'Cannot start threads: No default session is registered. Use '
          '`with sess.as_default()` or use explicit session in create_threads')

    if not isinstance(sess, session_lib.SessionInterface):
      if sess.__class__.__name__ in (
          'MonitoredSession', 'SingularMonitoredSession'):
        return []
      raise TypeError(
        f'sess must be a `tf.Session` object. Given class: {sess.__class__}')

    with sess.graph.as_default():
      threads = []
      for iterator in ops.get_collection(cls.__name__):
        threads.extend(iterator.create_threads(
          sess, coord=coord, daemon=True, start=True))
    return threads

  class Hook(SessionRunRewriting):
    r'''SessionRunHook that starts prefetching threads after session creation.
    '''
    def after_create_session(self, session, coord):
      Iterator.start(session, coord=coord)


SessionRunRewriting.register(
  Iterator.Hook, [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT])

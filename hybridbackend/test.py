#!/usr/bin/env python

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

r'''Module spawns up distributed training processes on each of the nodes.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json as _json
import multiprocessing as _mp
import os as _os
import socket as _socket
import sys as _sys


class Spawn(object):  # pylint: disable=useless-object-inheritance
  r'''Spawn multiple processes.
  '''
  class Barrier(object):  # pylint: disable=useless-object-inheritance
    r'''Barrier among python processes.
    '''
    def __init__(self, num_peers):
      self._num_peers = num_peers
      self._count = _mp.Value('i', 0)
      self._mutex = _mp.Semaphore(1)
      self._barrier = _mp.Semaphore(0)

    def wait(self):
      self._mutex.acquire()
      self._count.value += 1
      self._mutex.release()
      if self._count.value == self._num_peers:
        self._barrier.release()
      self._barrier.acquire()
      self._barrier.release()

  @classmethod
  def pick_unused_addr(cls, sock):
    sock.bind(('', 0))
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    return (f'{_socket.gethostbyname(_socket.gethostname())}:'
            f'{sock.getsockname()[1]}')

  def __init__(self, world_size=1):
    r'''Creates a group of processes.

    Args:
      world_size: Number of processes.
    '''
    self._world_size = world_size
    self._cluster = {'chief': []}
    self._socks = []
    self._socks.append(_socket.socket())
    self._cluster['chief'].append(self.pick_unused_addr(self._socks[0]))
    worker_procs = []
    for _ in range(world_size - 1):
      self._socks.append(_socket.socket())
      worker_procs.append(self.pick_unused_addr(self._socks[-1]))
    if worker_procs:
      self._cluster['worker'] = worker_procs

  def __del__(self):
    r'''Releases resources.
    '''
    for sock in self._socks:
      sock.close()

  def _target(self, fn, rank, barrier, collector):
    r'''Run function in a independent process.

    Args:
      fn: Function accepts a rank and returns a result.
      rank: Index of current process.
      barrier: Barrier between processes.
      collector: Results collector between processes.
    '''
    _os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    tf_config = {}
    tf_config['cluster'] = self._cluster
    tf_config['task'] = {}
    if rank == 0:
      tf_config['task']['type'] = 'chief'
      tf_config['task']['index'] = 0
    else:
      tf_config['task']['type'] = 'worker'
      tf_config['task']['index'] = rank - 1
    _os.environ['TF_CONFIG'] = _json.dumps(tf_config)
    result = fn(rank)
    collector.put((rank, result))
    barrier.wait()

  def __call__(self, fn, timeout_secs=300):
    r'''Run function in multiple processes.

    Args:
      fn: Fucntion returns a specification or ops to execute.
      timeout_secs: Seconds to wait processes.

    Returns:
      results of each process.
    '''
    barrier = self.Barrier(self._world_size)
    collector = _mp.Queue()
    procs = [
      _mp.Process(
        target=self._target,
        args=(fn, i, barrier, collector))
      for i in range(self._world_size)]
    for p in procs:
      p.daemon = True
      p.start()
    for p in procs:
      p.join(timeout_secs)
      p.terminate()
    results = {}
    for _ in range(self._world_size):
      item = collector.get()
      results[item[0]] = item[1]
    return results


def register(tags=None, extra=None):
  r'''Register tags for unit tests.
  '''
  if not isinstance(tags, (tuple, list)):
    tags = [tags]
  tags = [t.lower() for t in tags if t]
  allows = [t.lower() for t in _os.getenv('TEST_ALLOWS', '').split(',') if t]
  if allows:
    if not any((tag in allows for tag in tags)):
      print('The test is not allowed to run, skipped.', file=_sys.stderr)
      _sys.exit(0)
  denies = [t.lower() for t in _os.getenv('TEST_DENIES', '').split(',') if t]
  if denies:
    if any((tag in denies for tag in tags)):
      print('The test is denied to run, skipped.', file=_sys.stderr)
      _sys.exit(0)
  if extra:
    extras = [t.lower() for t in _os.getenv('TEST_EXTRAS', '').split(',') if t]
    if extra.lower() not in extras:
      print('The test is not needed to run, skipped.', file=_sys.stderr)
      _sys.exit(0)

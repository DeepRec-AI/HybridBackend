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

r'''Buffer of embeddings.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops

from hybridbackend.tensorflow.embedding.backend import EmbeddingBackend


class EmbeddingBuffer(object):  # pylint: disable=useless-object-inheritance
  r'''A buffer of embedding lookup.
  '''
  def __init__(self, column):
    r'''Constructs an embedding buffer.

    Args:
      column: Corresponding embedding column.
    '''
    self._impl = EmbeddingBackend.get()
    self._column = column
    self._sharded = self._impl.sharded(column)
    self._device = self._impl.device(column)

  def __call__(self, weight, ids):
    r'''Look up embeddings with buffering.
    '''
    with ops.device(self._device):
      return self._impl.lookup(
        self._column, weight, ids,
        sharded=self._sharded)

  @property
  def total_bytes(self):
    r'''Total bytes of the embedding buffer.
    '''
    return 0

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

r'''Functors for buffered embedding lookup.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops

from hybridbackend.tensorflow.feature_column.embedding_backend import \
    EmbeddingBackend
from hybridbackend.tensorflow.framework.context import Context


class EmbeddingLookupBuffered(object): # pylint: disable=useless-object-inheritance
  r'''Functor to lookup embeddings with buffering for multiple columns.

  Buffered embedding lookup is frequency-oblivious, which takes advantage of
  an accumulation buffer of gradients for reducing data transfer to the
  underlying storage. Compared to frequency-aware caching approach, this
  approach has better performance for worst cases.

  Also, due to high skewness of categorical inputs, high-frequent items have
  more chances to be buffered, and thus data transfer from the underlying
  storage might be reduced after some rounds too.
  '''
  def __init__(self, column_buckets):
    self._column_buckets = column_buckets
    self._impl = EmbeddingBackend.get()
    self._buffer_size = Context.get().param(
        'emb_buffer_size',
        default=0,
        env='HB_EMB_BUFFER_SIZE',
        parser=int)

  def _group_lookup_fallback(self, weight_buckets, input_shards_buckets):
    r'''Lookup embedding results without buffering.
    '''
    embedding_shards_buckets = []
    for bid, columns in enumerate(self._column_buckets):
      embedding_shards_list = []
      for idx, column in enumerate(columns):
        with ops.name_scope(column.name):
          embedding_shards = []
          for shard, _ in enumerate(Context.get().devices):
            embedding_shards.append(
                self._impl.lookup(
                    weight_buckets[bid][idx],
                    input_shards_buckets[bid][idx][shard]))
          embedding_shards_list.append(embedding_shards)
      embedding_shards_buckets.append(embedding_shards_list)
    return embedding_shards_buckets

  def __call__(self, weight_buckets, input_shards_buckets):
    r'''Lookup embedding results for multiple columns.
    '''
    if self._buffer_size > 0:
      raise NotImplementedError
    return self._group_lookup_fallback(weight_buckets, input_shards_buckets)

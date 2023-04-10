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

r'''Core functions for tabular datasets.

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from hybridbackend.tensorflow.data.dataframe import parse
from hybridbackend.tensorflow.data.deduplicate.dataset import deduplicate
from hybridbackend.tensorflow.data.rebatch.dataset import RebatchDataset


class TableFormats(object):  # pylint: disable=useless-object-inheritance
  r'''Table Formats.
  '''
  PARQUET = 11
  ORC = 21


class TabularDatasetCreator(object):  # pylint: disable=useless-object-inheritance
  r'''Creator for tabular datasets.
  '''
  def __init__(
      self, fn, filenames,
      fields=None,
      partition_count=1,
      partition_index=0,
      skip_corrupted_data=False,
      to_dense=False,
      num_parallel_reads=None,
      num_parallel_parser_calls=None,
      field_ignore_case=False,
      field_map_fn=None,
      **kwargs):
    r'''Create a `TabularDatasetCreator`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      skip_corrupted_data: (Optional.) If True, skip corrupted data.
      to_dense: (Optional.) If True, convert sparse tensors to dense. If
        it's a shape, then pad sparse tensors to specific shape.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_parallel_parser_calls: (Optional.) An integer representing the number
        of columns to parse in parallel. Defaults to parse columns sequentially.
        Note: All parquet datasets shares same parser pool, thus this argument
        can only be set once. if `tf.data.experimental.AUTOTUNE` is used, the
        number of parsers would be set for best performance.
      field_ignore_case: (Optional.) If True, ignore case of field names.
      field_map_fn: (Optional.) An function to transform fields.
    '''
    self._fn = fn
    self._filenames = filenames
    self._fields = fields if field_map_fn is None else field_map_fn(fields)
    self._partition_count = partition_count
    self._partition_index = partition_index
    self._skip_corrupted_data = skip_corrupted_data
    self._to_dense = to_dense
    self._field_ignore_case = field_ignore_case
    self._field_map_fn = field_map_fn
    self._key_idx_field_names = kwargs.pop('key_idx_field_names', None)
    self._value_field_names = kwargs.pop('value_field_names', None)

    if num_parallel_reads == dataset_ops.AUTOTUNE:
      if isinstance(filenames, (tuple, list)):
        num_parallel_reads = len(filenames)
      else:
        num_parallel_reads = 1
    if num_parallel_parser_calls is None:
      num_parallel_parser_calls = len(self._fields)
    if num_parallel_parser_calls == dataset_ops.AUTOTUNE:
      num_parallel_parser_calls = len(self._fields)
    if num_parallel_reads is not None and num_parallel_parser_calls is not None:
      arrow_num_threads = os.getenv('ARROW_NUM_THREADS', None)
      if arrow_num_threads is None:
        max_threads = os.cpu_count()
        if num_parallel_reads * num_parallel_parser_calls >= max_threads:
          num_parallel_reads = max(max_threads // num_parallel_parser_calls, 1)
        arrow_num_threads = num_parallel_reads * num_parallel_parser_calls
        os.environ['ARROW_NUM_THREADS'] = str(int(arrow_num_threads))
      else:
        arrow_num_threads = max(int(arrow_num_threads), 1)
        num_parallel_reads = arrow_num_threads // num_parallel_parser_calls
    if num_parallel_reads is not None and num_parallel_reads <= 1:
      num_parallel_reads = None
    self._num_parallel_reads = num_parallel_reads
    self._num_parallel_parser_calls = num_parallel_parser_calls

  @property
  def filenames(self):
    return self._filenames

  @property
  def fields(self):
    return self._fields

  @property
  def partition_count(self):
    return self._partition_count

  @property
  def partition_index(self):
    return self._partition_index

  @property
  def skip_corrupted_data(self):
    return self._skip_corrupted_data

  @property
  def to_dense(self):
    return self._to_dense

  @property
  def num_parallel_reads(self):
    return self._num_parallel_reads

  @property
  def num_parallel_parser_calls(self):
    return self._num_parallel_reads

  @property
  def field_ignore_case(self):
    return self._field_ignore_case

  @property
  def field_map_fn(self):
    return self._field_map_fn

  def _create_dataset(self, batch_size):
    r'''Create dataset for specific batch size in parallel.
    '''
    def _creator(filename):
      filename = ops.convert_to_tensor(filename, dtypes.string, name='filename')
      return self._fn(filename, batch_size)

    if self._num_parallel_reads == 1:
      return self._filenames.flat_map(_creator)
    if (self._num_parallel_reads is None
        or self._num_parallel_reads == dataset_ops.AUTOTUNE):
      return self._filenames.interleave(
        _creator, num_parallel_calls=self._num_parallel_reads)
    return readers.ParallelInterleaveDataset(
      self._filenames, _creator,
      cycle_length=self._num_parallel_reads,
      block_length=1,
      sloppy=True,
      buffer_output_elements=None,
      prefetch_input_elements=1)

  def read(self):
    r'''Read records in the dataset.
    '''
    ds = self._create_dataset(256)  # Prefetch 256 rows to speed up
    if (self._key_idx_field_names is not None
        and self._value_field_names is not None):
      ds = ds.apply(
        deduplicate(
          self._key_idx_field_names,
          self._value_field_names, fields=self._fields))
    ds = ds.apply(parse(pad=self._to_dense))
    return ds.unbatch()

  def batch(self, batch_size, drop_remainder=False):
    r'''Combines consecutive elements of this dataset into batches.

    The tensors in the resulting element will have an additional outer
    dimension, which will be `batch_size` (or `N % batch_size` for the last
    element if `batch_size` does not divide the number of input elements `N`
    evenly and `drop_remainder` is `False`). If your program depends on the
    batches having the same outer dimension, you should set the `drop_remainder`
    argument to `True` to prevent the smaller batch from being produced.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case its has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.
      allow_smaller: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether a batch could be smaller.

    Returns:
      Dataset: A `Dataset`.
    '''
    ds = self._create_dataset(batch_size)
    if (self._key_idx_field_names is not None
        and self._value_field_names is not None):
      ds = ds.apply(
        deduplicate(
          self._key_idx_field_names,
          self._value_field_names, fields=self._fields))
    ds = RebatchDataset(
      ds, self._fields, batch_size,
      drop_remainder=drop_remainder)
    return ds.apply(parse(pad=self._to_dense))

  def shuffle_batch(
      self, batch_size,
      drop_remainder=False,
      buffer_size=None,
      seed=None,
      reshuffle_each_iteration=None):
    r'''Randomly shuffles the elements of this dataset and then consecutive
    elements into batches.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case its has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.
      buffer_size: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        number of elements from this dataset from which the new
        dataset will sample. By default, buffer_size equals to batch_size.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        random seed that will be used to create the distribution. See
        `tf.set_random_seed` for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)

    Returns:
      Dataset: A `Dataset`.
    '''
    if buffer_size is None:
      buffer_size = batch_size
    ds = self._create_dataset(batch_size)
    if (self._key_idx_field_names is not None
        and self._value_field_names is not None):
      ds = ds.apply(
        deduplicate(
          self._key_idx_field_names,
          self._value_field_names, fields=self._fields))
    ds = RebatchDataset(
      ds, self._fields, batch_size,
      drop_remainder=drop_remainder,
      shuffle_buffer_size=buffer_size,
      shuffle_seed=seed,
      reshuffle_each_iteration=reshuffle_each_iteration)
    return ds.apply(parse(pad=self._to_dense))

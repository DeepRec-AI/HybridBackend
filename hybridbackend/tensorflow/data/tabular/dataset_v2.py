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

r'''Dataset that reads tabular data.

This class is compatible with TensorFlow 1.15.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.python.data.ops import dataset_ops

from hybridbackend.tensorflow.data.parquet.dataset_v2 import _ParquetDatasetV2
from hybridbackend.tensorflow.data.parquet.schema import \
  parquet_filenames_and_fields
from hybridbackend.tensorflow.data.tabular.core import TabularDatasetCreator


class TabularDatasetV2(dataset_ops.DatasetV2):  # pylint: disable=abstract-method
  r'''A Tabular Dataset that reads records.
  '''
  VERSION = 2002

  @classmethod
  def from_parquet(cls, filenames, **kwargs):
    return ParquetRecordDatasetV2(filenames, **kwargs)

  def __init__(
      self, filenames,
      fields=None,
      partition_count=1,
      partition_index=0,
      skip_corrupted_data=False,
      ignore_field_case=False,
      sparse_to_dense=False,
      num_parallel_reads=None,
      num_parallel_parser_calls=None):
    r'''Create a `TabularDataset`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      skip_corrupted_data: (Optional.) If True, skip corrupted data.
      ignore_field_case: (Optional.) If True, ignore case of field names.
      sparse_to_dense: (Optional.) If True, convert sparse tensors to dense. If
        it's a shape, then pad sparse tensors to specific shape.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_parallel_parser_calls: (Optional.) An integer representing the number
        of columns to parse in parallel. Defaults to parse columns sequentially.
        Note: All parquet datasets shares same parser pool, thus this argument
        can only be set once. if `tf.data.experimental.AUTOTUNE` is used, the
        number of parsers would be set for best performance.
    '''
    self._creator = TabularDatasetCreator(
      self._create_dataset, filenames,
      fields=fields,
      partition_count=partition_count,
      partition_index=partition_index,
      skip_corrupted_data=skip_corrupted_data,
      ignore_field_case=ignore_field_case,
      sparse_to_dense=sparse_to_dense,
      num_parallel_reads=num_parallel_reads,
      num_parallel_parser_calls=num_parallel_parser_calls)

    self._unbatched_ds = self._creator.read()
    super().__init__(self._unbatched_ds._variant_tensor)  # pylint: disable=protected-access

  def _inputs(self):
    return self._unbatched_ds._inputs()  # pylint: disable=protected-access

  @property
  def element_spec(self):
    return self._unbatched_ds.element_spec  # pylint: disable=protected-access

  @property
  def creator(self):
    return self._creator

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

    Returns:
      Dataset: A `Dataset`.
    '''
    return self._creator.batch(batch_size, drop_remainder=drop_remainder)

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
    return self._creator.shuffle_batch(
      batch_size, drop_remainder=drop_remainder,
      buffer_size=buffer_size,
      seed=seed,
      reshuffle_each_iteration=reshuffle_each_iteration)

  @abc.abstractmethod
  def _create_dataset(self, filename, batch_size):
    r'''Internal method to create a `TabularDataset`.
    '''


class ParquetRecordDatasetV2(TabularDatasetV2):
  r'''TabularDataset based Parquet reader.
  '''
  VERSION = 2002

  def __init__(
      self, filenames,
      fields=None,
      partition_count=1,
      partition_index=0,
      skip_corrupted_data=False,
      ignore_field_case=False,
      sparse_to_dense=False,
      num_parallel_reads=None,
      num_parallel_parser_calls=None):
    r'''Create a `TabularDataset`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      skip_corrupted_data: (Optional.) If True, skip corrupted data.
      ignore_field_case: (Optional.) If True, ignore case of field names.
      sparse_to_dense: (Optional.) If True, convert sparse tensors to dense. If
        it's a shape, then pad sparse tensors to specific shape.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_parallel_parser_calls: (Optional.) An integer representing the number
        of columns to parse in parallel. Defaults to parse columns sequentially.
        Note: All parquet datasets shares same parser pool, thus this argument
        can only be set once. if `tf.data.experimental.AUTOTUNE` is used, the
        number of parsers would be set for best performance.
    '''
    filenames, fields = parquet_filenames_and_fields(
      filenames, fields, lower=ignore_field_case)
    super().__init__(
      filenames,
      fields=fields,
      partition_count=partition_count,
      partition_index=partition_index,
      skip_corrupted_data=skip_corrupted_data,
      ignore_field_case=ignore_field_case,
      sparse_to_dense=sparse_to_dense,
      num_parallel_reads=num_parallel_reads,
      num_parallel_parser_calls=num_parallel_parser_calls)

  def _create_dataset(self, filename, batch_size):
    r'''Internal method to create a `TabularDataset`.
    '''
    return _ParquetDatasetV2(  # pylint: disable=abstract-class-instantiated
      filename, batch_size,
      fields=self.creator.fields,
      partition_count=self.creator.partition_count,
      partition_index=self.creator.partition_index,
      skip_corrupted_data=self.creator.skip_corrupted_data,
      drop_remainder=False)

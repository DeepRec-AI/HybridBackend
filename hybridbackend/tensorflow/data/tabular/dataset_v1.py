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

This class is compatible with TensorFlow 1.12.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated

from hybridbackend.libhybridbackend import orc_file_get_fields
from hybridbackend.libhybridbackend import parquet_file_get_fields
from hybridbackend.tensorflow.common import oplib as _ops
from hybridbackend.tensorflow.data.dataframe import build_fields
from hybridbackend.tensorflow.data.dataframe import build_filenames_and_fields
from hybridbackend.tensorflow.data.tabular.table import TableFormats
from hybridbackend.tensorflow.data.tabular.table import TabularDatasetCreator


class _TabularDatasetV1(dataset_ops.Dataset):
  r'''A Dataset that reads batches from tabular files.
  '''
  def __init__(
      self, fileformat, filename, batch_size, fields,
      partition_count=1,
      partition_index=0,
      skip_corrupted_data=False,
      drop_remainder=False):
    r'''Create a `ParquetDataset`.

    Args:
      fileformat: File format.
      filename: A 0-D `tf.string` tensor containing one filename.
      batch_size: Maxium number of samples in an output batch.
      fields: List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      skip_corrupted_data: (Optional.) If True, skip corrupted data.
      drop_remainder: (Optional.) If True, only keep batches with exactly
        `batch_size` samples.
    '''
    self._format = fileformat
    self._filename = ops.convert_to_tensor(
      filename, dtype=dtypes.string, name='filename')
    self._batch_size = ops.convert_to_tensor(
      batch_size, dtype=dtypes.int64, name='batch_size')
    self._fields = fields
    self._output_classes = {f.name: f.output_classes for f in self._fields}
    self._output_types = {f.name: f.output_types for f in self._fields}
    self._output_shapes = {f.name: f.output_shapes for f in self._fields}
    self._field_names = nest.flatten({f.name: f.name for f in self._fields})
    self._field_dtypes = nest.flatten({f.name: f.dtype for f in self._fields})
    self._field_ragged_ranks = nest.flatten(
      {f.name: f.ragged_rank for f in self._fields})
    self._field_shapes = nest.flatten({f.name: f.shape for f in self._fields})
    self._partition_count = partition_count
    self._partition_index = partition_index
    self._skip_corrupted_data = skip_corrupted_data
    self._drop_remainder = drop_remainder
    super().__init__()

  def _as_variant_tensor(self):
    return _ops.hb_tabular_dataset(
      self._filename,
      self._batch_size,
      format=self._format,
      field_names=self._field_names,
      field_dtypes=self._field_dtypes,
      field_ragged_ranks=self._field_ragged_ranks,
      field_shapes=self._field_shapes,
      partition_count=self._partition_count,
      partition_index=self._partition_index,
      skip_corrupted_data=self._skip_corrupted_data,
      drop_remainder=self._drop_remainder)

  def _inputs(self):
    return []

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_types(self):
    return self._output_types

  @property
  def output_shapes(self):
    return self._output_shapes


class TabularDatasetV1(dataset_ops.Dataset):
  r'''A Tabular Dataset that reads records.
  '''
  VERSION = 2001

  @classmethod
  def from_parquet(cls, filenames, **kwargs):
    return ParquetRecordDatasetV1(filenames, **kwargs)

  @classmethod
  def from_orc(cls, filenames, **kwargs):
    return OrcRecordDatasetV1(filenames, **kwargs)

  def __init__(
      self, filenames,
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
    r'''Create a `TabularDataset`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of partitions.
      partition_index: (Optional.) Index of partitions.
      skip_corrupted_data: (Optional.) If True, skip corrupted data.
      to_dense: (Optional.) If True, convert sparse tensors to dense. If
        it's a shape, then pad sparse tensors to specific shape.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_parallel_parser_calls: (Optional.) An integer representing the number
        of columns to parse in parallel. Defaults to parse columns sequentially.
        Note: All tabular datasets shares same parser pool, thus this argument
        can only be set once. if `tf.data.experimental.AUTOTUNE` is used, the
        number of parsers would be set for best performance.
      field_ignore_case: (Optional.) If True, ignore case of field names.
      field_map_fn: (Optional.) An function to transform fields.
    '''
    self._creator = TabularDatasetCreator(
      self._create_dataset, filenames,
      fields=fields,
      partition_count=partition_count,
      partition_index=partition_index,
      skip_corrupted_data=skip_corrupted_data,
      to_dense=to_dense,
      num_parallel_reads=num_parallel_reads,
      num_parallel_parser_calls=num_parallel_parser_calls,
      field_ignore_case=field_ignore_case,
      field_map_fn=field_map_fn,
      **kwargs)

    self._unbatched_ds = self._creator.read()
    super().__init__()

  def _as_variant_tensor(self):
    return self._unbatched_ds._as_variant_tensor()  # pylint: disable=protected-access

  def _inputs(self):
    return self._unbatched_ds._inputs()  # pylint: disable=protected-access

  @property
  def output_shapes(self):
    return self._unbatched_ds.output_shapes

  @property
  def output_types(self):
    return self._unbatched_ds.output_types

  @property
  def output_classes(self):
    return self._unbatched_ds.output_classes

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


class ParquetRecordDatasetV1(TabularDatasetV1):
  r'''TabularDataset based Parquet reader.
  '''
  VERSION = 2001

  def __init__(
      self, filenames,
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
    r'''Create a `TabularDataset`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of partitions.
      partition_index: (Optional.) Index of partitions.
      skip_corrupted_data: (Optional.) If True, skip corrupted data.
      to_dense: (Optional.) If True, convert sparse tensors to dense. If
        it's a shape, then pad sparse tensors to specific shape.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_parallel_parser_calls: (Optional.) An integer representing the number
        of columns to parse in parallel. Defaults to parse columns sequentially.
        Note: All tabular datasets shares same parser pool, thus this argument
        can only be set once. if `tf.data.experimental.AUTOTUNE` is used, the
        number of parsers would be set for best performance.
      field_ignore_case: (Optional.) If True, ignore case of field names.
      field_map_fn: (Optional.) An function to transform fields.
    '''
    filenames, fields = build_filenames_and_fields(
      filenames, parquet_file_get_fields, fields, lower=field_ignore_case)
    self._fields = fields
    super().__init__(
      filenames,
      fields=fields,
      partition_count=partition_count,
      partition_index=partition_index,
      skip_corrupted_data=skip_corrupted_data,
      to_dense=to_dense,
      num_parallel_reads=num_parallel_reads,
      num_parallel_parser_calls=num_parallel_parser_calls,
      field_ignore_case=field_ignore_case,
      field_map_fn=field_map_fn,
      **kwargs)

  @property
  def fields(self):
    return self._fields

  def _create_dataset(self, filename, batch_size):
    r'''Internal method to create a `TabularDataset`.
    '''
    return _TabularDatasetV1(  # pylint: disable=abstract-class-instantiated
      TableFormats.PARQUET, filename, batch_size,
      fields=self.creator.fields,
      partition_count=self.creator.partition_count,
      partition_index=self.creator.partition_index,
      skip_corrupted_data=self.creator.skip_corrupted_data,
      drop_remainder=False)


class OrcRecordDatasetV1(TabularDatasetV1):
  r'''TabularDataset based ORC reader.
  '''
  VERSION = 2001

  def __init__(
      self, filenames,
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
    r'''Create a `TabularDataset`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of partitions.
      partition_index: (Optional.) Index of partitions.
      skip_corrupted_data: (Optional.) If True, skip corrupted data.
      to_dense: (Optional.) If True, convert sparse tensors to dense. If
        it's a shape, then pad sparse tensors to specific shape.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_parallel_parser_calls: (Optional.) An integer representing the number
        of columns to parse in parallel. Defaults to parse columns sequentially.
        Note: All tabular datasets shares same parser pool, thus this argument
        can only be set once. if `tf.data.experimental.AUTOTUNE` is used, the
        number of parsers would be set for best performance.
      field_ignore_case: (Optional.) If True, ignore case of field names.
      field_map_fn: (Optional.) An function to transform fields.
    '''
    filenames, fields = build_filenames_and_fields(
      filenames, orc_file_get_fields, fields, lower=field_ignore_case)
    self._fields = fields
    super().__init__(
      filenames,
      fields=fields,
      partition_count=partition_count,
      partition_index=partition_index,
      skip_corrupted_data=skip_corrupted_data,
      to_dense=to_dense,
      num_parallel_reads=num_parallel_reads,
      num_parallel_parser_calls=num_parallel_parser_calls,
      field_ignore_case=field_ignore_case,
      field_map_fn=field_map_fn,
      **kwargs)

  @property
  def fields(self):
    return self._fields

  def _create_dataset(self, filename, batch_size):
    r'''Internal method to create a `TabularDataset`.
    '''
    return _TabularDatasetV1(  # pylint: disable=abstract-class-instantiated
      TableFormats.ORC, filename, batch_size,
      fields=self.creator.fields,
      partition_count=self.creator.partition_count,
      partition_index=self.creator.partition_index,
      skip_corrupted_data=self.creator.skip_corrupted_data,
      drop_remainder=False)


class ParquetDatasetV1(dataset_ops.Dataset):
  r'''A Parquet Dataset that reads batches from parquet files.
  '''
  VERSION = 2001

  @classmethod
  @deprecated(None, 'Prefer hb.data.Dataset.from_parquet instead.')
  def read_schema(cls, filename, fields=None, lower=False):
    r'''Read schema from a parquet file.

    Args:
      filename: Path of the parquet file.
      fields: Existing field definitions or field names.
      lower: Convert field name to lower case if not found.

    Returns:
      Field definition list.
    '''
    return build_fields(filename, parquet_file_get_fields, fields, lower=lower)

  @deprecated(None, 'Prefer hb.data.Dataset.from_parquet instead.')
  def __init__(
      self, filenames,
      batch_size=1,
      fields=None,
      partition_count=1,
      partition_index=0,
      skip_corrupted_data=False,
      drop_remainder=False,
      num_parallel_reads=None,
      num_sequential_reads=1,
      num_parallel_parser_calls=None):
    r'''Create a `ParquetDataset`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      batch_size: (Optional.) Maxium number of samples in an output batch.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      skip_corrupted_data: (Optional.) If True, skip corrupted data.
      drop_remainder: (Optional.) If True, only keep batches with exactly
        `batch_size` samples.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_sequential_reads: (Optional.) A `tf.int64` scalar representing the
        number of batches to read in sequential. Defaults to 1.
      num_parallel_parser_calls: (Optional.) An integer representing the number
        of columns to parse in parallel. Defaults to parse columns sequentially.
        Note: All parquet datasets shares same parser pool, thus this argument
        can only be set once. if `tf.data.experimental.AUTOTUNE` is used, the
        number of parsers would be set for best performance.
    '''
    filenames, self._fields = build_filenames_and_fields(
      filenames, parquet_file_get_fields, fields)
    self._partition_count = partition_count
    self._partition_index = partition_index
    self._skip_corrupted_data = skip_corrupted_data
    self._drop_remainder = drop_remainder

    if num_parallel_reads == dataset_ops.AUTOTUNE:
      if isinstance(filenames, (tuple, list)):
        num_parallel_reads = len(filenames)
      else:
        num_parallel_reads = 1
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

    def _create_dataset(f):
      f = ops.convert_to_tensor(f, dtypes.string, name='filename')
      return _TabularDatasetV1(
        TableFormats.PARQUET, f, batch_size,
        fields=self._fields,
        partition_count=self._partition_count,
        partition_index=self._partition_index,
        skip_corrupted_data=self._skip_corrupted_data,
        drop_remainder=self._drop_remainder)
    self._impl = self._build_dataset(
      _create_dataset, filenames, num_parallel_reads, num_sequential_reads)
    super().__init__()

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
  def drop_remainder(self):
    return self._drop_remainder

  def _as_variant_tensor(self):
    return self._impl._as_variant_tensor()  # pylint: disable=protected-access

  def _inputs(self):
    return self._impl._inputs()  # pylint: disable=protected-access

  @property
  def output_shapes(self):
    return self._impl.output_shapes

  @property
  def output_types(self):
    return self._impl.output_types

  @property
  def output_classes(self):
    return self._impl.output_classes

  def _build_dataset(
      self, dataset_creator, filenames,
      num_parallel_reads=None, num_sequential_reads=1):
    r'''Internal method to create a `ParquetDataset`.
    '''
    if num_parallel_reads is None:
      return filenames.flat_map(dataset_creator)
    return filenames.interleave(
      dataset_creator,
      cycle_length=num_parallel_reads if num_parallel_reads > 0 else 1,
      block_length=num_sequential_reads,
      num_parallel_calls=num_parallel_reads)

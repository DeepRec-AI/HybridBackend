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

r'''Dataset that reads Parquet files.

This class is compatible with TensorFlow 1.12.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import nest

from hybridbackend.tensorflow.common import oplib as _ops
from hybridbackend.tensorflow.data.parquet.schema import parquet_fields
from hybridbackend.tensorflow.data.parquet.schema import \
  parquet_filenames_and_fields


class _ParquetDatasetV1(dataset_ops.Dataset):
  r'''A Parquet Dataset that reads batches from parquet files.
  '''

  def __init__(
      self, filename, batch_size, fields,
      partition_count=1,
      partition_index=0,
      drop_remainder=False):
    r'''Create a `ParquetDataset`.

    Args:
      filename: A 0-D `tf.string` tensor containing one filename.
      batch_size: Maxium number of samples in an output batch.
      fields: List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      drop_remainder: (Optional.) If True, only keep batches with exactly
        `batch_size` samples.
    '''
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
    self._drop_remainder = drop_remainder
    super().__init__()

  def _as_variant_tensor(self):
    return _ops.hb_parquet_tabular_dataset(
      self._filename,
      self._batch_size,
      field_names=self._field_names,
      field_dtypes=self._field_dtypes,
      field_ragged_ranks=self._field_ragged_ranks,
      field_shapes=self._field_shapes,
      partition_count=self._partition_count,
      partition_index=self._partition_index,
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


class ParquetDatasetV1(dataset_ops.Dataset):
  r'''A Parquet Dataset that reads batches from parquet files.
  '''
  VERSION = 2001

  @classmethod
  def read_schema(cls, filename, fields=None, lower=False):
    r'''Read schema from a parquet file.

    Args:
      filename: Path of the parquet file.
      fields: Existing field definitions or field names.
      lower: Convert field name to lower case if not found.

    Returns:
      Field definition list.
    '''
    return parquet_fields(filename, fields, lower=lower)

  def __init__(
      self, filenames,
      batch_size=1,
      fields=None,
      partition_count=1,
      partition_index=0,
      drop_remainder=False,
      num_parallel_reads=None,
      num_sequential_reads=1):
    r'''Create a `ParquetDataset`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      batch_size: (Optional.) Maxium number of samples in an output batch.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      drop_remainder: (Optional.) If True, only keep batches with exactly
        `batch_size` samples.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_sequential_reads: (Optional.) A `tf.int64` scalar representing the
        number of batches to read in sequential. Defaults to 1.
    '''
    filenames, self._fields = parquet_filenames_and_fields(filenames, fields)
    self._partition_count = partition_count
    self._partition_index = partition_index
    self._drop_remainder = drop_remainder

    def _create_dataset(f):
      f = ops.convert_to_tensor(f, dtypes.string, name='filename')
      return _ParquetDatasetV1(
        f, batch_size,
        fields=self._fields,
        partition_count=self._partition_count,
        partition_index=self._partition_index,
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

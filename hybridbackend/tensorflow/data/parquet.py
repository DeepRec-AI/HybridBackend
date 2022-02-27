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

r'''Parquet related utilities.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six import string_types as string

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging

from hybridbackend.tensorflow.data.dataframe import DataFrame
from hybridbackend import libhybridbackend as _lib


def parquet_fields(filename, fields=None):
  r'''Get fields from a parquet file.

  Args:
    filename: Path of the parquet file.
    fields: Existing field definitions or field names.

  Returns:
    Field definitions.
  '''
  logging.info(f'Reading fields from {filename} ...')
  all_field_tuples = _lib.parquet_file_get_fields(filename)  # pylint: disable=c-extension-no-member
  if not all_field_tuples:
    raise ValueError(
      f'No supported fields found in parquet file {filename}')
  all_fields = {
    f[0]: {'dtype': f[1], 'ragged_rank': f[2]}
    for f in all_field_tuples}
  if fields is None:
    fields = all_fields.keys()
  fields = tuple(fields)
  new_fields = []
  for f in fields:
    if isinstance(f, DataFrame.Field):
      new_fields.append(f)
      continue
    if not isinstance(f, string):
      raise ValueError(
        f'Field {f} is not a DataFrame.Field or a string')
    if f not in all_fields:
      raise ValueError(
        f'Field {f} is not found in the parquet file {filename}')
    new_fields.append(DataFrame.Field(
      f,
      dtype=np.dtype(all_fields[f]['dtype']),
      ragged_rank=all_fields[f]['ragged_rank'],
      shape=[None]))
  return tuple(new_fields)


def parquet_filenames_and_fields(filenames, fields):
  r'''Check and fetch parquet filenames and fields.

  Args:
    filenames: List of Path of parquet file list.
    fields: Existing field definitions or field names.

  Returns:
    Validated file names and fields.
  '''
  if isinstance(filenames, string):
    filenames = [filenames]
    fields = parquet_fields(filenames[0], fields=fields)
  elif isinstance(filenames, (tuple, list)):
    for f in filenames:
      if not isinstance(f, string):
        raise ValueError(f'{f} in `filenames` must be a string')
    fields = parquet_fields(filenames[0], fields=fields)
  elif isinstance(filenames, dataset_ops.Dataset):
    if filenames.output_types != dtypes.string:
      raise TypeError(
        '`filenames` must be a `tf.data.Dataset` of `tf.string` elements.')
    if not filenames.output_shapes.is_compatible_with(
        tensor_shape.TensorShape([])):
      raise ValueError(
        '`filenames` must be a `tf.data.Dataset` of scalar `tf.string` '
        'elements.')
    if fields is None:
      raise ValueError('`fields` must be specified.')
    if not isinstance(fields, (tuple, list)):
      raise ValueError('`fields` must be a list of `hb.data.DataFrame.Field`.')
    for f in fields:
      if not isinstance(f, DataFrame.Field):
        raise ValueError(f'{f} must be `hb.data.DataFrame.Field`.')
  elif isinstance(filenames, ops.Tensor):
    if filenames.dtype != dtypes.string:
      raise TypeError(
        '`filenames` must be a `tf.Tensor` of `tf.string`.')
    if fields is None:
      raise ValueError('`fields` must be specified.')
    if not isinstance(fields, (tuple, list)):
      raise ValueError('`fields` must be a list of `hb.data.DataFrame.Field`.')
    for f in fields:
      if not isinstance(f, DataFrame.Field):
        raise ValueError(f'{f} must be `hb.data.DataFrame.Field`.')
  else:
    raise ValueError(
      f'`filenames` {filenames} must be a `tf.data.Dataset` of scalar '
      '`tf.string` elements or can be converted to a `tf.Tensor` of '
      '`tf.string`.')

  if not isinstance(filenames, dataset_ops.Dataset):
    filenames = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    filenames = array_ops.reshape(filenames, [-1], name='filenames')
    filenames = dataset_ops.Dataset.from_tensor_slices(filenames)
  return filenames, fields

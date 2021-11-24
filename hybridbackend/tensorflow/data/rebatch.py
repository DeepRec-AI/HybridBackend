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

r'''Rebatching related utilities.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hybridbackend.tensorflow.data.dataframe import DataFrame


def input_fields(input_dataset, fields=None):
  r'''Fetch and validate fields from input dataset.
  '''
  if fields is None:
    ds = input_dataset
    while ds:
      if hasattr(ds, 'fields'):
        fields = ds.fields
        break
      if not hasattr(ds, '_input_dataset'):
        break
      ds = ds._input_dataset  # pylint: disable=protected-access
  if not fields:
    raise ValueError('`fields` must be specified')
  if not isinstance(fields, (tuple, list)):
    raise ValueError('`fields` must be a list of `hb.data.DataFrame.Field`.')
  for f in fields:
    if not isinstance(f, DataFrame.Field):
      raise ValueError('{} must be `hb.data.DataFrame.Field`.'.format(f))
  return fields

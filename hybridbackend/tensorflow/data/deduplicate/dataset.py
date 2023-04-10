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

r'''Dataset that compresses DataFrame values.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import nest

from hybridbackend.tensorflow.data.dataframe import input_fields


def deduplicate(
    key_idx_field_names,
    value_field_names,
    fields=None):
  r'''Deduplicate fields specified in `value_field_names`
    by using specified fields in `key_field_names`.

  Args:
    key_idx_field_names: A list of string as names of fields utilized to
      recover the key fields.
    value_field_names: A List of list of string as fields to be
      deduplicated by key fields.
    fields: (Optional) fields of dataset.
  '''
  def _apply_fn(dataset):
    all_fields = input_fields(dataset, fields=fields)
    all_field_names = nest.flatten({f.name: f.name for f in all_fields})
    map_name_to_fields = {f.name: f for f in all_fields}

    for key_idx_field_name in key_idx_field_names:
      if key_idx_field_name not in all_field_names:
        raise ValueError(
          f'Key idx Field {key_idx_field_name} must be within the Fields')

    if len(value_field_names) != len(key_idx_field_names):
      raise ValueError(
        'Value field names must have the same length as key idx field names')

    key_idx_field_to_value_fields = {}
    for i, name in enumerate(key_idx_field_names):
      key_idx_field_to_value_fields[name] = value_field_names[i]

    for k, v_list in key_idx_field_to_value_fields.items():
      for v in v_list:
        if v not in all_field_names:
          raise ValueError(
            f'Value Field {v} must be within the Fields')
        map_name_to_fields[v].set_restore_idx_field(map_name_to_fields[k])
    return dataset
  return _apply_fn

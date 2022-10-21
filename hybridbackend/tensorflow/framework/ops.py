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

r'''Classes and functions used to construct graphs.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.util import nest


class GraphKeys(object):  # pylint: disable=useless-object-inheritance
  r'''Names to use for graph collections.
  '''
  # Collection for trainable variables placed at every devices.
  TRAINABLE_REPLICATED = 'trainable_replicated'
  # Collection for trainable variables placed at multiple devices.
  TRAINABLE_SHARDED = 'trainable_sharded'
  # Collection for variables placed at multiple devices.
  SHARDED_VARIABLES = 'sharded_variables'
  # Collection for resources placed at multiple devices.
  SHARDED_RESOURCES = 'sharded_resources'
  # Collection for resources or variables should not be replicated.
  NOT_REPLICATED = 'not_replicated'


class ModeKeys(object):  # pylint: disable=useless-object-inheritance
  r'''Keys to use for modes.
  '''
  TRAIN = 'train'
  EVAL = 'eval'
  PREDICT = 'infer'


class MultiValues(object):  # pylint: disable=useless-object-inheritance
  r'''Multiple values.
  '''
  def __init__(self, items):
    r'''Creates a MultiValues.
    '''
    if not isinstance(items, dict):
      raise ValueError('items should be a dict')
    self._items = items

  @classmethod
  def select(cls, values, key):
    r'''Select value on specific key.

    Args:
      values: A structure contains MultiValues instance.
      key: A key to select.

    Returns:
      A structure contains values on specific key.
    '''
    return nest.map_structure(
      lambda v: v[key] if isinstance(v, MultiValues) else v, values)

  @classmethod
  def build(cls, values):
    r'''Build a MultiValues from values.

    Args:
      values: A structure of values.

    Returns:
      A MultiValues.
    '''
    if isinstance(values, MultiValues):
      return values

    if isinstance(values, dict):
      return MultiValues(dict(values))

    if isinstance(values, (tuple, list)):
      return MultiValues(dict(enumerate(values)))

    return MultiValues({0: values})

  @classmethod
  def build_from(cls, keys, fn, *args, **kwargs):
    r'''Create a MultiValues from function.

    Args:
      fn: function accepts *args, **kwargs on each key.
      args: arguments of fn.
      kwargs: key-value arguments of fn.

    Returns:
      A MultiValues as result of fn on each index.
    '''
    results = {}
    for key in sorted(keys):
      with ops.name_scope(f'{key}'):
        results[key] = fn(
          *MultiValues.select(args, key),
          **MultiValues.select(kwargs, key))
    return MultiValues(results)

  def __getitem__(self, key):
    return self._items[key]

  def __len__(self):
    return len(self._items)

  def __str__(self):
    return 'MultiValues ' + str(self._items)

  def __repr__(self):
    return 'MultiValues ' + str(self._items)

  @property
  def keys(self):
    r'''Tuple of keys.
    '''
    return tuple(self._items.keys())

  @property
  def values(self):
    r'''Tuple of values.
    '''
    return tuple(self._items.values())

  @property
  def items(self):
    r'''Internal key to value dict.
    '''
    return self._items

  def map(self, fn, *args, **kwargs):
    r'''Call fn on each value.
    Args:
      fn: old_value, *args, **kwargs -> new_value.
      args: arguments of fn, which contains MultiValues.
      kwargs: key-value arguments of fn, which contains MultiValues.

    Returns:
      A mapped MultiValues instance.
    '''
    results = {}
    for key in sorted(self._items.keys()):
      with ops.name_scope(f'{key}'):
        results[key] = fn(
          self._items[key],
          *MultiValues.select(args, key),
          **MultiValues.select(kwargs, key))
    return MultiValues(results)

  def check(self, fn):
    r'''Check whether fn to all items is True.
    Args:
      fn: value -> True or False.

    Returns:
      True or False.
    '''
    return all(set(self.map(fn)))

  def regroup(self):
    r'''Transform to structures of MultiValues.
    '''
    if not self._items:
      return None

    first_value = next(iter(self._items.values()))
    if len(self._items) == 1:
      return first_value

    keys = sorted(self._items.keys())
    flatten_values = [nest.flatten(self._items[k]) for k in keys]
    regrouped_values = [
      MultiValues({k: t[i] for i, k in enumerate(keys)})
      for t in zip(*flatten_values)]
    return nest.pack_sequence_as(first_value, regrouped_values)

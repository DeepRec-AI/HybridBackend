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

r'''Classes and functions for options.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework import dtypes


class Options(object):  # pylint: disable=useless-object-inheritance
  r'''Options for configuration.
  '''
  _builders = {}

  @classmethod
  def register_builder(cls, builder):
    cls._builders[builder.TYPE] = builder

  @classmethod
  def has_builder(cls, value):
    return type(value) in cls._builders

  @classmethod
  def clone(cls, value, default):
    r'''Clone value from default.
    '''
    dtype = type(default) if default is not None else None
    if dtype in cls._builders:
      return cls._builders[dtype].clone(value, default)
    return value

  @classmethod
  def parse(cls, value, default):
    r'''Parse value for specific default.
    '''
    dtype = type(default) if default is not None else None
    if dtype is None or value is None or isinstance(value, dtype):
      return value
    if dtype not in cls._builders:
      return dtype(value)
    return cls._builders[dtype].parse(value, default)

  def __init__(self):
    self.__dict__['__defaults__'] = {}
    self.__dict__['__items__'] = {}

  def __getattr__(self, name):
    if name not in self.__dict__['__items__']:
      raise AttributeError(name)
    return self.__dict__['__items__'][name]

  def __setattr__(self, name, value):
    if name not in self.__dict__['__items__']:
      raise AttributeError(name)
    value = Options.clone(value, self.__dict__['__defaults__'][name])
    self.__dict__['__items__'][name] = value

  def __str__(self):
    return str(self.__dict__['__items__'])

  def update(self, **opts):
    r'''Update options using key-value arguments.
    Args:
      opts: options dict need to modify.
    Returns:
      Options dict before modification.
    '''
    prev_opts = {
      k: self.__dict__['__items__'][k]
      for k in opts if k in self.__dict__['__items__']}
    for k, v in opts.items():
      self.__setattr__(k, v)
    return prev_opts

  def register(self, name, value, env=None):
    r'''If the option does not exist, insert the option with specified value.

    Args:
      name: Option name.
      value: Default value for specific option.
      env: Environment variable to fetch the default value.

    Returns:
      Options object.
    '''
    if name in self.__dict__['__items__']:
      return self

    if env is None:
      self.__dict__['__defaults__'][name] = value
      self.__dict__['__items__'][name] = value
      return self

    env_value = os.getenv(env, value)
    env_value = Options.parse(env_value, value)
    self.__dict__['__defaults__'][name] = env_value
    self.__dict__['__items__'][name] = env_value
    return self


class OptionBuilder(object):  # pylint: disable=useless-object-inheritance
  r'''Option builder for specific type.
  '''
  TYPE = type(None)

  def clone(self, value, default):
    r'''Create a new value from specific default value.
    '''
    if (value is not None
        and default is not None
        and not isinstance(value, type(default))):
      raise ValueError(f'{value} should be a {type(default)}')
    return value

  def parse(self, value, default):
    r'''Parse from value for specific default value.
    '''
    raise NotImplementedError


class BoolOptionBuilder(OptionBuilder):  # pylint: disable=useless-object-inheritance
  r'''Option builder for bool.
  '''
  TYPE = bool

  def parse(self, value, default):
    r'''Parse from value for specific default value.
    '''
    del default
    if value is None:
      return None
    if isinstance(value, bool):
      return value
    trues = ['TRUE', 'YES', '1']
    falses = ['FALSE', 'NO', '0']
    if value.upper() in trues:
      return True
    if value.upper() in falses:
      return False
    return bool(int(value))


Options.register_builder(BoolOptionBuilder())


class DTypeOptionBuilder(OptionBuilder):  # pylint: disable=useless-object-inheritance
  r'''Option builder for DType.
  '''
  TYPE = dtypes.DType

  def parse(self, value, default):
    r'''Parse from value for specific default value.
    '''
    del default
    return dtypes.as_dtype(value)


Options.register_builder(DTypeOptionBuilder())


class SelectorOption(object):  # pylint: disable=useless-object-inheritance
  r'''Option with selector.
  '''
  def __init__(self, default, items=None):
    self.__items__ = dict(items) if items else {}
    self.__items__['*'] = default

  def __getitem__(self, name):
    if name not in self.__items__:
      if '*' in self.__items__:
        return self.__items__['*']
      raise AttributeError(name)
    return self.__items__[name]

  def __str__(self):
    return str(self.__items__)

  @property
  def default(self):
    return self.__items__['*']

  def as_dict(self):
    return dict(self.__items__)


class SelectorOptionBuilder(OptionBuilder):  # pylint: disable=useless-object-inheritance
  r'''Option builder for feature column options.
  '''
  TYPE = SelectorOption

  def clone(self, value, default):
    r'''Create a new value from specific default value.
    '''
    if isinstance(value, SelectorOption):
      return value

    if isinstance(value, dict):
      return SelectorOption(default.default, dict(value))

    if (default.default is not None
        and not isinstance(value, type(default.default))):
      raise ValueError(
        f'{value} should be hb.SelectorOption '
        f'or dict or {type(default.default)}')
    items = default.as_dict()
    items['*'] = value
    return SelectorOption(value, items)

  def parse(self, value, default):
    r'''Parse from value for specific default value.
    '''
    return self.clone(Options.parse(value, default.default), default)


Options.register_builder(SelectorOptionBuilder())

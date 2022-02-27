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


class Options(object):  # pylint: disable=useless-object-inheritance
  r'''Options for configuration.
  '''
  def __init__(self, **kwargs):
    self.__items__ = dict(kwargs)

  def __getattr__(self, attr):
    if attr not in self.__items__:
      raise AttributeError(attr)
    return self.__items__[attr]

  def __str__(self):
    return str(self.__items__)

  def get(self, key, default_value):
    return self.__items__.get(key, default_value)

  def update(self, key, value):
    self.__items__[key] = value

  @property
  def items(self):
    return dict(self.__items__)

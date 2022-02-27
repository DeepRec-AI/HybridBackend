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


class CollectiveOps(object): # pylint: disable=useless-object-inheritance
  r'''Collective operations.
  '''
  SUM = 0
  PROD = 1
  MAX = 2
  MIN = 3
  AVG = 4


class GraphKeys(object):  # pylint: disable=useless-object-inheritance
  r'''Names to use for graph collections.
  '''
  # Collection for variables placed at multiple devices.
  SHARDED_VARIABLES = "sharded_variables"
  # Collection for resources placed at multiple devices.
  SHARDED_RESOURCES = "sharded_resources"

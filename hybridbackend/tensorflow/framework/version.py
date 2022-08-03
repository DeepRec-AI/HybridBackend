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

r'''Version related utilities.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.version

_TENSORFLOW_VERSION = None


def tf_version():
  r'''Get tensorflow version.
  '''
  global _TENSORFLOW_VERSION
  if _TENSORFLOW_VERSION:
    return _TENSORFLOW_VERSION
  try:
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    _TENSORFLOW_VERSION = distutils.version.LooseVersion(tf.VERSION)
  except ImportError as imp:
    _TENSORFLOW_VERSION = None
    raise ImportError('Tensorflow version is not supported') from imp
  return _TENSORFLOW_VERSION


def tf_version_check(ver):
  r'''Whether tensorflow version is greater than ver.
  '''
  return tf_version() >= distutils.version.LooseVersion(ver)

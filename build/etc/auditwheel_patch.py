#!/usr/bin/env python

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

r'''Patch manylinux-policy.json for auditwheel.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

LIB_WHITELIST = [
  'libcrypt.so.1',
  'libnccl.so.2',
  'libtensorflow_framework.so.1',
  'libhybridbackend.so',
  'libhybridbackend_tensorflow.so']

if __name__ == '__main__':
  auditwheel_home = os.path.dirname(os.path.dirname(__file__))
  policy_file_path = os.path.join(
    auditwheel_home,
    'lib/python3.9/site-packages/auditwheel/policy/manylinux-policy.json')
  with open(policy_file_path, encoding='utf8') as f:
    policies = json.load(f)
  for p in policies:
    p['lib_whitelist'].extend(LIB_WHITELIST)
  with open(policy_file_path, 'w', encoding='utf8') as f:
    json.dump(policies, f)

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

r'''Communicators and distribution options.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hybridbackend.tensorflow.distribute.communicator import CollectiveOps \
  as ops
from hybridbackend.tensorflow.distribute.communicator import Communicator
from hybridbackend.tensorflow.distribute.communicator_pool import \
  CommunicatorPool
from hybridbackend.tensorflow.distribute.nccl import NcclCommunicator
from hybridbackend.tensorflow.framework.context import Context as _ctx


_ = (
  _ctx.get().options
  .register('comm_default', 'NCCL', env='HB_COMM_DEFAULT')
  .register('comm_pool_name', 'default')
  .register('comm_pool_capacity', 1)
  .register('comm_pubsub_device', '/cpu:0'))

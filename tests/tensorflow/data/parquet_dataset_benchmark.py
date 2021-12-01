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

r'''Parquet dataset Benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import monitored_session

from hybridbackend.tensorflow.data import make_one_shot_iterator
from hybridbackend.tensorflow.data import ParquetDataset
from hybridbackend.tensorflow.training import StepStatHook


# pylint: disable=missing-docstring
def benchmark(params):
  with ops.Graph().as_default():
    ds = ParquetDataset(
        [params.filename] * params.epochs,
        batch_size=params.batch_size)
    batch = make_one_shot_iterator(ds).get_next()
    count_op = array_ops.shape(list(batch.values())[0])[0]
    train_op = control_flow_ops.group(batch.values())
    with monitored_session.MonitoredTrainingSession(
        is_chief=True,
        hooks=[StepStatHook(count_op=count_op)]) as sess:
      while not sess.should_stop():
        sess.run(train_op)


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  os.environ['MALLOC_CONF'] = 'background_thread:true,metadata_thp:auto'
  os.environ['ARROW_NUM_THREADS'] = '16'
  logging.set_verbosity(logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=200000)
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument('filename')
  benchmark(parser.parse_args())

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

r'''TFRecord dataset benchmark.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.data.ops.readers import TFRecordDataset
from tensorflow.python.training import monitored_session

from hybridbackend.tensorflow.data import make_one_shot_iterator
from hybridbackend.tensorflow.training import StepStatHook


# pylint: disable=missing-docstring
def describe():
  return {
      'col%d' % c: parsing_ops.FixedLenFeature([1], dtypes.int64)
      for c in range(200)}


def benchmark(params):
  with ops.Graph().as_default():
    ds = TFRecordDataset([params.filename])
    ds = ds.batch(params.batch_size)
    ds = ds.map(lambda line: parsing_ops.parse_example(line, describe()))
    batch = make_one_shot_iterator(ds).get_next()
    count_op = array_ops.shape(batch[0])[0]
    train_op = control_flow_ops.group(batch)
    with monitored_session.MonitoredTrainingSession(
        is_chief=True,
        hooks=[StepStatHook(count_op=count_op)]) as sess:
      while not sess.should_stop():
        sess.run(train_op)


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  logging.set_verbosity(logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=200000)
  parser.add_argument('filename')
  benchmark(parser.parse_args())

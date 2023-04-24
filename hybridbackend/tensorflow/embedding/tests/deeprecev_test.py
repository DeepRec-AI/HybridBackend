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

r'''Tests for embedding columns upon DeepRec EV.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import hybridbackend.common.test as hbtest

# pylint: disable=missing-docstring
# pylint: disable=import-outside-toplevel


def _test_get_embedding_variable(_):
  import tensorflow as tf

  import hybridbackend.tensorflow as hb

  with tf.Graph().as_default():
    with hb.scope():
      with hb.embedding_scope():
        with tf.device('/cpu:0'):
          var = tf.get_embedding_variable(
            'var_1',
            embedding_dim=3,
            initializer=tf.ones_initializer(tf.float32),
            partitioner=tf.fixed_size_partitioner(num_shards=4))
        emb = tf.nn.embedding_lookup(
          var, tf.cast([0, 1, 2, 5, 6, -7], tf.int64))
      fun = tf.multiply(emb, 2.0, name='multiply')
      loss = tf.reduce_sum(fun, name='reduce_sum')
      opt = tf.train.FtrlOptimizer(
        0.1,
        l1_regularization_strength=2.0,
        l2_regularization_strength=0.00001)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      with tf.train.MonitoredTrainingSession('') as sess:
        emb_result, loss_result, _ = sess.run([emb, loss, train_op])
        return (emb_result, loss_result)


@unittest.skipUnless(
  (os.getenv('HYBRIDBACKEND_WITH_CUDA') == 'ON'
   and os.getenv('HYBRIDBACKEND_WITH_TENSORFLOW_DISTRO') == '99881015'),
  'DeepRec on GPU required')
@unittest.skipUnless(
  os.getenv('HYBRIDBACKEND_WITH_NCCL') == 'ON', 'NCCL required')
class DeepRecEVTest(unittest.TestCase):
  '''Tests for embedding column.
  '''
  def setUp(self):  # pylint: disable=invalid-name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

  def test_get_embedding_variable(self):
    results = hbtest.Spawn()(_test_get_embedding_variable)
    print(results)

  def test_get_embedding_variable_2g(self):
    results = hbtest.Spawn(2)(_test_get_embedding_variable)
    print(results)


# pylint: enable=missing-docstring
if __name__ == '__main__':
  hbtest.main(f'{__file__}.xml')

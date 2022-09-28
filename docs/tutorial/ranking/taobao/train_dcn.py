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

r'''Training model based on stacked DCNv2 over taobao dataset.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from ranking.data import DataSpec
from ranking.model import stacked_dcn_v2
import tensorflow as tf
from tensorflow.python.util import module_wrapper as _deprecation  # pylint: disable=protected-access

import hybridbackend.tensorflow as hb

_deprecation._PER_MODULE_WARNING_LIMIT = 0  # pylint: disable=protected-access
tf.get_logger().propagate = False


def main(args):
  if args.disable_optimization:
    hb.disable_optimization()

  data_spec = DataSpec.read(args.data_spec_filename)

  with tf.device('/cpu:0'):
    ds = hb.data.ParquetDataset(
      args.filenames,
      batch_size=args.train_batch_size,
      num_parallel_reads=len(args.filenames),
      num_parallel_parser_calls=args.num_parsers,
      drop_remainder=True)
    ds = ds.apply(hb.data.parse())
    ds = ds.prefetch(args.num_prefetches)
    iterator = tf.data.make_one_shot_iterator(ds)

  with tf.device(args.fe_device):
    iterator = hb.data.Iterator(iterator, args.num_prefetches)
    batch = iterator.get_next()
    batch.pop('ts')
    labels = tf.reshape(tf.to_float(batch.pop('label')), shape=[-1, 1])
    features = []
    for f in batch:
      feature = batch[f]
      if data_spec.embedding_dims[f] is None:
        features.append(data_spec.transform_numeric(f, feature))
      else:
        with tf.device(args.embedding_weight_device):
          if args.use_ev:
            embedding_weights = tf.get_embedding_variable(
              f'{f}_weight',
              key_dtype=feature.dtype,
              embedding_dim=data_spec.embedding_dims[f],
              initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
          else:
            embedding_weights = tf.get_variable(
              f'{f}_weight',
              shape=(data_spec.embedding_sizes[f], data_spec.embedding_dims[f]),
              initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
        features.append(
          data_spec.transform_categorical(f, feature, embedding_weights))

  with tf.device(args.dnn_device):
    logits = stacked_dcn_v2(features, args.mlp_dims)
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
    step = tf.train.get_or_create_global_step()
    opt = tf.train.AdagradOptimizer(learning_rate=args.lr)
    train_op = opt.minimize(loss, global_step=step)

  hooks = []
  hooks.append(hb.data.Iterator.Hook())
  hooks.append(
    hb.train.StepStatHook(count=args.train_batch_size, unit='sample'))
  if args.log_every_n_iter is not None:
    hooks.append(
      tf.train.LoggingTensorHook(
        {'step': step, 'loss': loss},
        every_n_iter=args.log_every_n_iter))
  if args.train_max_steps is not None:
    hooks.append(tf.train.StopAtStepHook(args.train_max_steps))
  chief_only_hooks = []
  if args.profile_every_n_iter is not None:
    chief_only_hooks.append(
      tf.train.ProfilerHook(
        save_steps=args.profile_every_n_iter,
        output_dir='.'))
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  config.gpu_options.force_gpu_compatible = True
  with tf.train.MonitoredTrainingSession(
      '',
      hooks=hooks,
      chief_only_hooks=chief_only_hooks,
      config=config) as sess:
    while not sess.should_stop():
      sess.run(train_op)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--disable-optimization', default=False, action='store_true')
  parser.add_argument(
    '--use-ev', default=False, action='store_true')
  parser.add_argument('--num-parsers', type=int, default=16)
  parser.add_argument('--num-prefetches', type=int, default=2)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--fe-device', default='/gpu:0')
  parser.add_argument('--dnn-device', default='/gpu:0')
  parser.add_argument('--embedding-weight-device', default='/cpu:0')
  parser.add_argument(
    '--mlp-dims', nargs='+', default=[1024, 1024, 512, 256, 1])
  parser.add_argument('--train-batch-size', type=int, default=16000)
  parser.add_argument('--train-max-steps', type=int, default=None)
  parser.add_argument('--log-every-n-iter', type=int, default=10)
  parser.add_argument('--profile-every-n-iter', type=int, default=None)
  parser.add_argument(
    '--data-spec-filename', default='ranking/taobao/data/spec.json')
  parser.add_argument('filenames', nargs='+')
  main(parser.parse_args())

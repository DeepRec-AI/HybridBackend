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

r'''Training and evaluating model over taobao dataset.
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


class RankingModel:
  r'''Ranking model.
  '''
  def __init__(self, args):
    self._args = args

  def input_dataset(self, filenames, batch_size):
    r'''Get input dataset.
    '''
    with tf.device('/cpu:0'):
      ds = hb.data.Dataset.from_parquet(filenames)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.prefetch(self._args.num_prefetches)
      return ds

  def read(self, ds):
    r'''Read from dataset.
    '''
    with tf.device(self._args.transform_device):
      iterator = tf.data.make_one_shot_iterator(ds)
      inputs = iterator.get_next()
      inputs.pop('ts')
      labels = tf.reshape(tf.to_float(inputs.pop('label')), shape=[-1, 1])
      return inputs, labels

  def extract_features(self, inputs):
    r'''Extract features.
    '''
    with tf.device(self._args.transform_device):
      features = []
      for f in inputs:
        feature = inputs[f]
        if self._args.data_spec.embedding_dims[f] is None:
          features.append(self._args.data_spec.transform_numeric(f, feature))
        else:
          with hb.embedding_scope():
            with tf.device(self._args.embedding_weight_device):
              if self._args.use_ev:
                with tf.variable_scope(
                    f'{f}_embedding',
                    partitioner=tf.fixed_size_partitioner(
                      hb.context.world_size)):
                  embedding_weights = tf.get_embedding_variable(
                    f'{f}_weight',
                    key_dtype=feature.dtype,
                    embedding_dim=self._args.data_spec.embedding_dims[f],
                    initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
              else:
                embedding_weights = tf.get_variable(
                  f'{f}_weight',
                  shape=(
                    self._args.data_spec.embedding_sizes[f],
                    self._args.data_spec.embedding_dims[f]),
                  initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
          features.append(
            self._args.data_spec.transform_categorical(
              f, feature, embedding_weights))
      return features

  def compute_loss(self, logits, labels):
    r'''Compute loss.
    '''
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))

  def train(self, filenames):
    r'''Train model.
    '''
    train_dataset = self.input_dataset(filenames, self._args.train_batch_size)
    inputs, labels = self.read(train_dataset)
    features = self.extract_features(inputs)
    logits = stacked_dcn_v2(features, self._args.mlp_dims)
    loss = self.compute_loss(logits, labels)
    step = tf.train.get_or_create_global_step()
    opt = tf.train.AdagradOptimizer(learning_rate=self._args.lr)
    train_op = opt.minimize(loss, global_step=step)
    return step, loss, train_op

  def evaluate(self, filenames):
    r'''Evaluate model.
    '''
    eval_dataset = self.input_dataset(filenames, self._args.eval_batch_size)
    inputs, labels = self.read(eval_dataset)
    features = self.extract_features(inputs)
    logits = stacked_dcn_v2(features, self._args.mlp_dims)
    loss = self.compute_loss(logits, labels)
    return {
      'auc': hb.metrics.auc(
        labels=labels,
        predictions=logits,
        name='eval_auc'),
      'loss': (loss, tf.no_op())}

  def predict(self):
    r'''Predict model.
    '''
    inputs = self._args.data_spec.build_placeholders()
    features = self.extract_features(inputs)
    logits = stacked_dcn_v2(features, self._args.mlp_dims)
    return tf.saved_model.predict_signature_def(inputs, {'score': logits})


def main(args):
  if len(args.filenames) > 1:
    train_filenames = args.filenames[:-1]
    eval_filenames = args.filenames[-1]
  else:
    train_filenames = args.filenames
    eval_filenames = args.filenames
  model = RankingModel(args)
  step, loss, train_op = model.train(train_filenames)

  hooks = []
  if args.eval_every_n_iter is not None:
    hooks.append(hb.train.EvaluationHook(
      lambda: model.evaluate(eval_filenames),
      every_n_iter=args.eval_every_n_iter,
      summary_dir=args.output_dir))
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
        output_dir=args.output_dir))
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  config.gpu_options.force_gpu_compatible = True
  with tf.train.MonitoredTrainingSession(
      '',
      hooks=hooks,
      chief_only_hooks=chief_only_hooks,
      checkpoint_dir=args.output_dir,
      config=config) as sess:
    while not sess.should_stop():
      sess.run(train_op)

  hb.train.export(
    args.savedmodel_dir,
    tf.train.latest_checkpoint(args.output_dir),
    model.predict)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--use-ev', default=False, action='store_true')
  parser.add_argument('--num-prefetches', type=int, default=2)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--transform-device', default='/gpu:0')
  parser.add_argument('--embedding-weight-device', default='/cpu:0')
  parser.add_argument(
    '--mlp-dims', nargs='+', default=[1024, 1024, 512, 256, 1])
  parser.add_argument('--train-batch-size', type=int, default=16000)
  parser.add_argument('--train-max-steps', type=int, default=None)
  parser.add_argument('--eval-batch-size', type=int, default=100)
  parser.add_argument('--eval-every-n-iter', type=int, default=50)
  parser.add_argument('--log-every-n-iter', type=int, default=10)
  parser.add_argument('--profile-every-n-iter', type=int, default=None)
  parser.add_argument('--output-dir', default='./outputs')
  parser.add_argument('--savedmodel-dir', default='./outputs/savedmodels')
  parser.add_argument(
    '--data-spec-filename', default='ranking/taobao/data/spec.json')
  parser.add_argument('filenames', nargs='+')
  parsed = parser.parse_args()
  parsed.data_spec = DataSpec.read(parsed.data_spec_filename)
  with hb.scope():
    main(parsed)

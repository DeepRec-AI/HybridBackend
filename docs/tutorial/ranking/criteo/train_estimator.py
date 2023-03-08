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

r'''Training and evaluating model over criteo dataset using Estimator API.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from ranking.data import DataSpec
from ranking.model import dlrm
from ranking.optimization import sgd_decay_optimize
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
      ds = ds.map(
        lambda batch: (
          {f: batch[f] for f in batch if f not in ('label')},
          batch['label']))
      ds = ds.prefetch(self._args.num_prefetches)
      return ds

  def input_receiver(self):
    r'''Prediction input receiver.
    '''
    inputs = self._args.data_spec.build_placeholders()
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

  def call(self, features, labels, mode, params):
    r'''Model function for estimator.
    '''
    del params
    numeric_fields = [
      fname for fname in features
      if self._args.data_spec.embedding_dims[fname] is None]
    categorical_fields = [
      fname for fname in features
      if self._args.data_spec.embedding_dims[fname] is not None]
    numeric_features = [
      self._args.data_spec.transform_numeric(f, features[f])
      for f in numeric_fields]
    if self._args.use_shared_emb:
      embedding_columns = tf.feature_column.shared_embedding_columns(
        [tf.feature_column.categorical_column_with_identity(
          key=f, num_buckets=self._args.data_spec.embedding_sizes[f],
          default_value=self._args.data_spec.defaults[f])
          for f in categorical_fields],
        dimension=self._args.embedding_dim,
        initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
    else:
      embedding_columns = [
        tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
            key=f,
            num_buckets=self._args.data_spec.embedding_sizes[f],
            default_value=self._args.data_spec.defaults[f]),
          dimension=self._args.embedding_dim,
          initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
        for f in categorical_fields]
    with hb.embedding_scope(), tf.device('/cpu:0'):
      categorical_features = [
        tf.feature_column.input_layer(features, [c])
        for c in embedding_columns]
    logits = dlrm(
      numeric_features, categorical_features,
      self._args.bottom_mlp_dims,
      self._args.embedding_dim,
      self._args.top_mlp_dims)

    if mode == tf.estimator.ModeKeys.TRAIN:
      labels = tf.reshape(tf.to_float(labels), shape=[-1, 1])
      loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
      train_op = sgd_decay_optimize(
        loss,
        lr_initial_value=self._args.lr_initial_value,
        lr_warmup_steps=self._args.lr_warmup_steps,
        lr_decay_start_step=self._args.lr_decay_start_step,
        lr_decay_steps=self._args.lr_decay_steps)
      chief_only_hooks = []
      if self._args.profile_every_n_iter is not None:
        chief_only_hooks.append(
          tf.train.ProfilerHook(
            save_steps=self._args.profile_every_n_iter,
            output_dir=self._args.output_dir))
      return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_chief_hooks=chief_only_hooks)

    if mode == tf.estimator.ModeKeys.EVAL:
      labels = tf.reshape(tf.to_float(labels), shape=[-1, 1])
      loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
      return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops={
          'auc': hb.metrics.auc(labels, logits, name='eval_auc')})

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={'score': logits})

    return None


def main(args):
  if len(args.filenames) > 1:
    train_filenames = args.filenames[:-1]
    eval_filenames = args.filenames[-1]
  else:
    train_filenames = args.filenames
    eval_filenames = args.filenames
  model = RankingModel(args)

  estimator = hb.estimator.Estimator(
    model.call,
    model_dir=args.output_dir)

  if args.evaluate:
    estimator.evaluate(
      input_fn=lambda: model.input_dataset(
        eval_filenames, args.eval_batch_size),
      steps=args.eval_max_steps)
  elif args.predict:
    pred_result = estimator.predict(
      input_fn=lambda: model.input_dataset(
        eval_filenames, args.pred_batch_size),
      predict_keys=['score'],
      yield_single_examples=False)
    print(next(pred_result))
  else:
    estimator.train_and_evaluate(
      tf.estimator.TrainSpec(
        input_fn=lambda: model.input_dataset(
          train_filenames, args.train_batch_size),
        max_steps=args.train_max_steps),
      tf.estimator.EvalSpec(
        input_fn=lambda: model.input_dataset(
          eval_filenames, args.eval_batch_size)),
      eval_every_n_iter=args.eval_every_n_iter)
    estimator.export_saved_model(
      args.savedmodel_dir,
      model.input_receiver)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--disable-imputation', default=False, action='store_true')
  parser.add_argument(
    '--use-ev', default=False, action='store_true')
  parser.add_argument(
    '--use-shared-emb', default=False, action='store_true')
  parser.add_argument('--num-parsers', type=int, default=16)
  parser.add_argument('--num-prefetches', type=int, default=2)
  parser.add_argument('--lr-initial-value', type=float, default=24.)
  parser.add_argument('--lr-warmup-steps', type=int, default=2750)
  parser.add_argument('--lr-decay-start-step', type=int, default=48000)
  parser.add_argument('--lr-decay-steps', type=int, default=27772)
  parser.add_argument('--transform-device', default='/gpu:0')
  parser.add_argument('--embedding-weight-device', default='/gpu:0')
  parser.add_argument('--embedding-dim', type=int, default=16)
  parser.add_argument(
    '--bottom-mlp-dims', nargs='+', default=[512, 256])
  parser.add_argument(
    '--top-mlp-dims', nargs='+', default=[1024, 1024, 512, 256, 1])
  parser.add_argument(
    '--mlp-dims', nargs='+', default=[1024, 1024, 512, 256, 1])
  parser.add_argument('--train-batch-size', type=int, default=64000)
  parser.add_argument('--train-max-steps', type=int, default=None)
  parser.add_argument('--evaluate', default=False, action='store_true')
  parser.add_argument('--eval-batch-size', type=int, default=100)
  parser.add_argument('--eval-max-steps', type=int, default=10)
  parser.add_argument('--predict', default=False, action='store_true')
  parser.add_argument('--pred-batch-size', type=int, default=10)
  parser.add_argument('--eval-every-n-iter', type=int, default=50)
  parser.add_argument('--log-every-n-iter', type=int, default=10)
  parser.add_argument('--profile-every-n-iter', type=int, default=None)
  parser.add_argument('--output-dir', default='./outputs')
  parser.add_argument('--savedmodel-dir', default='./outputs/savedmodels')
  parser.add_argument(
    '--data-spec-filename', default='ranking/criteo/data/spec.json')
  parser.add_argument('filenames', nargs='+')
  parsed = parser.parse_args()
  parsed.data_spec = DataSpec.read(
    parsed.data_spec_filename,
    disable_imputation=parsed.disable_imputation,
    disable_transform=True,
    override_embedding_size=parsed.embedding_dim)
  hb.enable_optimization(relocate_ops=True)
  main(parsed)

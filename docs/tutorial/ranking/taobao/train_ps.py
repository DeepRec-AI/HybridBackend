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

r'''Training and evaluating model over taobao dataset using Estimator API.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

from ranking.data import DataSpec
from ranking.model import stacked_dcn_v2
import tensorflow as tf
from tensorflow.python.util import module_wrapper as _deprecation  # pylint: disable=protected-access

import hybridbackend.tensorflow as hb

hb.enable_optimization(relocate_ops=True)

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
          {f: batch[f] for f in batch if f not in ('ts', 'label')},
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
      fs.name for fs in self._args.data_spec.feature_specs
      if fs.name not in ('ts', 'label') and fs.embedding is None]
    wide_features = [
      self._args.data_spec.transform_numeric(f, features[f])
      for f in numeric_fields]
    categorical_fields = [
      fs.name for fs in self._args.data_spec.feature_specs
      if fs.name not in ('ts', 'label') and fs.embedding is not None]
    embedding_columns = [
      tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
          key=f,
          num_buckets=self._args.data_spec.embedding_sizes[f],
          default_value=self._args.data_spec.defaults[f]),
        dimension=self._args.data_spec.embedding_dims[f],
        initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
      for f in categorical_fields]
    deep_features = [
      tf.feature_column.input_layer(features, [c])
      for c in embedding_columns]
    logits = stacked_dcn_v2(wide_features + deep_features, self._args.mlp_dims)

    if mode == tf.estimator.ModeKeys.TRAIN:
      labels = tf.reshape(tf.to_float(labels), shape=[-1, 1])
      loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
      step = tf.train.get_or_create_global_step()
      opt = tf.train.AdagradOptimizer(learning_rate=self._args.lr)
      opt = hb.train.SyncReplicasOptimizer(
        opt, replicas_to_aggregate=len(self._args.world))
      train_op = opt.minimize(loss, global_step=step)
      hooks = [opt.make_session_run_hook(self._args.rank == 0, num_tokens=0)]
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
        training_hooks=hooks,
        training_chief_hooks=chief_only_hooks)

    if mode == tf.estimator.ModeKeys.EVAL:
      labels = tf.reshape(tf.to_float(labels), shape=[-1, 1])
      loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
      return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops={
          'auc': tf.metrics.auc(labels, logits, name='eval_auc')})

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

  estimator = tf.estimator.Estimator(
    model.call,
    model_dir=args.output_dir)
  tf.estimator.train_and_evaluate(
    estimator,
    tf.estimator.TrainSpec(
      input_fn=lambda: model.input_dataset(
        train_filenames, args.train_batch_size),
      max_steps=args.train_max_steps),
    tf.estimator.EvalSpec(
      input_fn=lambda: model.input_dataset(
        eval_filenames, args.eval_batch_size)))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--use-ev', default=False, action='store_true')
  parser.add_argument('--num-prefetches', type=int, default=2)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument(
    '--mlp-dims', nargs='+', default=[1024, 1024, 512, 256, 1])
  parser.add_argument('--train-batch-size', type=int, default=16000)
  parser.add_argument('--train-max-steps', type=int, default=None)
  parser.add_argument('--eval-batch-size', type=int, default=100)
  parser.add_argument('--profile-every-n-iter', type=int, default=None)
  parser.add_argument('--output-dir', default='./outputs')
  parser.add_argument('--savedmodel-dir', default='./outputs/savedmodels')
  parser.add_argument(
    '--data-spec-filename', default='ranking/taobao/data/spec.json')
  parser.add_argument('filenames', nargs='+')
  parsed = parser.parse_args()
  parsed.data_spec = DataSpec.read(parsed.data_spec_filename)
  tf_config = json.loads(os.getenv('TF_CONFIG', '{}'))
  if not tf_config:
    raise ValueError('TF_CONFIG must be set')
  cluster = tf_config['cluster']
  num_chiefs = len(cluster['chief'])
  num_workers = len(cluster['worker'])
  world = [f'/job:chief/task:{t}' for t in range(num_chiefs)]
  world += [f'/job:worker/task:{t}' for t in range(num_workers)]
  parsed.world = world
  task = tf_config['task']
  task_type = task['type']
  task_index = int(task['index'])
  current_device = f'/job:{task_type}/task:{task_index}'
  try:
    parsed.rank = world.index(current_device)
  except:  # pylint: disable=bare-except
    parsed.rank = -1
  main(parsed)

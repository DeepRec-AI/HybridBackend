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

r'''Training and evaluating model over taobao dataset using Keras API.
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


class RankingModel(tf.layers.Layer):
  r'''Ranking model.
  '''
  def __init__(self, args, trainable=True, name=None, **kwargs):
    super().__init__(trainable=trainable, name=name, **kwargs)
    self._args = args
    self._numeric_fields = [
      fs.name for fs in self._args.data_spec.feature_specs
      if fs.name not in ('ts', 'label') and fs.embedding is None]
    self._categorical_fields = [
      fs.name for fs in self._args.data_spec.feature_specs
      if fs.name not in ('ts', 'label') and fs.embedding is not None]
    self._embedding_columns = [
      tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
          key=f,
          num_buckets=self._args.data_spec.embedding_sizes[f],
          default_value=self._args.data_spec.defaults[f]),
        dimension=self._args.data_spec.embedding_dims[f],
        initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
      for f in self._categorical_fields]

  def call(self, features, labels=None, **kwargs):
    r'''Model function for estimator.
    '''
    del kwargs

    wide_features = [
      self._args.data_spec.transform_numeric(f, features[f])
      for f in self._numeric_fields]
    cols_to_output_tensors = {}
    tf.keras.layers.DenseFeatures(
      self._embedding_columns)(
        {f: features[f] for f in self._categorical_fields},
        cols_to_output_tensors=cols_to_output_tensors)
    deep_features = [
      cols_to_output_tensors[f] for f in self._embedding_columns]
    logits = stacked_dcn_v2(wide_features + deep_features, self._args.mlp_dims)
    if labels is None:
      return logits
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
    return loss


def input_dataset(args, filenames, batch_size):
  r'''Get input dataset.
  '''
  with tf.device('/cpu:0'):
    ds = hb.data.Dataset.from_parquet(filenames)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(
      lambda batch: (
        {f: batch[f] for f in batch if f not in ('ts', 'label')},
        tf.reshape(batch['label'], shape=[-1, 1])))
    ds = ds.prefetch(args.num_prefetches)
    return ds


def predict_fn(args):
  r'''Predict function.
  '''
  inputs_numeric = {
    fs.name: tf.placeholder(dtype=tf.float32)
    for fs in args.data_spec.feature_specs if fs.name not in ('ts', 'label')
    and fs.embedding is None}

  inputs_categorical = {
    fs.name: tf.placeholder(dtype=tf.int32, shape=[None])
    for fs in args.data_spec.feature_specs if fs.name not in ('ts', 'label')
    and fs.embedding is not None}

  inputs = {}
  inputs.update(inputs_numeric)
  inputs.update(inputs_categorical)

  predict_logits = RankingModel(args)(inputs)
  outputs = {'score': predict_logits}
  return tf.saved_model.predict_signature_def(inputs, outputs)


def main(args):
  if len(args.filenames) > 1:
    train_filenames = args.filenames[:-1]
    val_filenames = args.filenames[-1]
  else:
    train_filenames = args.filenames
    val_filenames = args.filenames

  train_dataset = input_dataset(
    args, train_filenames, args.train_batch_size)
  features, labels = tf.data.make_one_shot_iterator(train_dataset).get_next()

  val_dataset = input_dataset(
    args, val_filenames, args.eval_batch_size)

  model_output = RankingModel(args)(features)
  dcnv2_in_keras = tf.keras.Model(inputs=[features], outputs=model_output)

  def loss_func(y_true, y_pred):
    return tf.reduce_mean(
      tf.keras.losses.binary_crossentropy(y_true, y_pred))
  opt = tf.train.AdagradOptimizer(learning_rate=args.lr)

  if args.weights_dir is not None:
    dcnv2_in_keras.load_weights(args.weights_dir)

  dcnv2_in_keras.compile(
    loss=loss_func,
    metrics=[tf.keras.metrics.AUC()],
    optimizer=opt,
    target_tensors=labels,
    clipnorm=1.0,
    clipvalue=1.0)

  if args.evaluate:
    dcnv2_in_keras.evaluate(
      x=None,
      y=None,
      batch_size=args.eval_batch_size,
      checkpoint_dir=args.output_dir,
      steps=args.eval_max_steps)
  else:
    dcnv2_in_keras.fit(
      x=None,
      y=None,
      epochs=1,
      validation_data=val_dataset,
      batch_size=args.train_batch_size,
      validation_steps=args.eval_max_steps,
      steps_per_epoch=args.train_max_steps,
      checkpoint_dir=args.output_dir,
      keep_checkpoint_max=2,
      monitor='val_auc',
      mode='max',
      save_best_only=True)

    dcnv2_in_keras.export_saved_model(
      args.output_dir,
      lambda: predict_fn(args))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--use-ev', default=False, action='store_true')
  parser.add_argument('--num-prefetches', type=int, default=2)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--extract-features-device', default='/gpu:0')
  parser.add_argument('--embedding-weight-device', default='/cpu:0')
  parser.add_argument(
    '--mlp-dims', nargs='+', default=[1024, 1024, 512, 256, 1])
  parser.add_argument('--train-batch-size', type=int, default=16000)
  parser.add_argument('--train-max-steps', type=int, default=None)
  parser.add_argument('--evaluate', default=False, action='store_true')
  parser.add_argument('--eval-batch-size', type=int, default=100)
  parser.add_argument('--eval-max-steps', type=int, default=1)
  parser.add_argument('--eval-every-n-iter', type=int, default=50)
  parser.add_argument('--log-every-n-iter', type=int, default=10)
  parser.add_argument('--profile-every-n-iter', type=int, default=None)
  parser.add_argument('--output-dir', default='./outputs')
  parser.add_argument('--weights-dir', default=None)
  parser.add_argument('--savedmodel-dir', default='./outputs/savedmodels')
  parser.add_argument(
    '--data-spec-filename', default='ranking/taobao/data/spec.json')
  parser.add_argument('filenames', nargs='+')
  parsed = parser.parse_args()
  parsed.data_spec = DataSpec.read(parsed.data_spec_filename)
  with hb.scope():
    main(parsed)

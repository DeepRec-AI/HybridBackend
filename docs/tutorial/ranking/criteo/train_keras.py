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

r'''Training and evaluating model over criteo dataset using Keras API.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from ranking.data import DataSpec
from ranking.model import dlrm
from ranking.optimization import lr_with_linear_warmup_and_polynomial_decay
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import \
  feature_column as contrib_fc
from tensorflow.contrib.layers.python.layers import \
  feature_column_ops as contrib_fc_ops
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
      if fs.name not in ('label')
      and self._args.data_spec.embedding_dims[fs.name] is None]
    self._categorical_fields = [
      fs.name for fs in self._args.data_spec.feature_specs
      if fs.name not in ('label')
      and self._args.data_spec.embedding_dims[fs.name] is not None]
    if self._args.use_shared_emb_feats is not None:
      shared_emb_feats = self._args.use_shared_emb_feats.split(',')
    else:
      shared_emb_feats = []
    non_shared_categorical_fields = [
      f for f in self._categorical_fields
      if f not in shared_emb_feats]
    emb_var_names_in_ckpt = {}
    if self._args.load_ckpt_dir is not None:
      restoreable_saveables, _ = zip(
        *tf.train.list_variables(self._args.load_ckpt_dir))
      for f in self._categorical_fields:
        for vname in list(restoreable_saveables):
          if f in vname:
            emb_var_names_in_ckpt[f] = vname
            break
          emb_var_names_in_ckpt[f] = None
    else:
      emb_var_names_in_ckpt = {f: None for f in self._categorical_fields}

    self._embedding_columns = [
      contrib_fc.embedding_column(
        contrib_fc.sparse_column_with_integerized_feature(
          f,
          self._args.data_spec.embedding_sizes[f]),
        dimension=self._args.embedding_dim,
        initializer=tf.random_uniform_initializer(-1e-3, 1e-3),
        ckpt_to_load_from=self._args.load_ckpt_dir,
        tensor_name_in_ckpt=emb_var_names_in_ckpt[f])
      for f in non_shared_categorical_fields]
    if shared_emb_feats:
      self._embedding_columns.extend(
        contrib_fc.shared_embedding_columns(
          [contrib_fc.sparse_column_with_integerized_feature(
            f,
            self._args.data_spec.embedding_sizes[f])
            for f in shared_emb_feats],
          dimension=self._args.embedding_dim,
          initializer=tf.random_uniform_initializer(-1e-3, 1e-3),
          ckpt_to_load_from=self._args.load_ckpt_dir,
          tensor_name_in_ckpt=emb_var_names_in_ckpt[shared_emb_feats[0]]))

  def call(self, features, labels=None, **kwargs):
    r'''Model function for estimator.
    '''
    del kwargs

    numeric_features = [
      self._args.data_spec.transform_numeric(f, features[f])
      for f in self._numeric_fields]
    cols_to_output_tensors = {}
    contrib_fc_ops.input_from_feature_columns(
      {f: features[f] for f in self._categorical_fields},
      self._embedding_columns,
      cols_to_outs=cols_to_output_tensors)
    categorical_features = [
      cols_to_output_tensors[f] for f in self._embedding_columns]
    logits = dlrm(
      numeric_features, categorical_features,
      self._args.bottom_mlp_dims,
      self._args.embedding_dim,
      self._args.top_mlp_dims)
    if labels is None:
      return logits
    loss = tf.reduce_mean(
      tf.keras.losses.binary_crossentropy(labels, logits))
    return loss


class DlrmInKeras(hb.keras.Model):
  r'''Wraps the model by using keras.Model API'''
  def __init__(self, args):
    super().__init__()
    self._args = args
    self.dlrm = RankingModel(args)

  def call(self, inputs):  # pylint: disable=method-hidden
    return self.dlrm(inputs)

  def input_dataset(self, filenames, batch_size):
    r'''Get input data
    '''
    with tf.device('/cpu:0'):
      ds = hb.data.Dataset.from_parquet(filenames)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.map(
        lambda batch: (
          {f: batch[f] for f in batch if f not in ('label')},
          tf.reshape(batch['label'], shape=[-1, 1])))
      ds = ds.prefetch(self._args.num_prefetches)
      return ds


def predict_fn(args):
  r'''Predict function.
  '''
  inputs_numeric = {
    fs.name: tf.placeholder(dtype=tf.float32)
    for fs in args.data_spec.feature_specs if fs.name not in ('label')
    and args.data_spec.embedding_dims[fs.name] is None}

  inputs_categorical = {
    fs.name: tf.placeholder(dtype=tf.int32, shape=[None])
    for fs in args.data_spec.feature_specs if fs.name not in ('label')
    and args.data_spec.embedding_dims[fs.name] is not None}

  inputs = {}
  inputs.update(inputs_numeric)
  inputs.update(inputs_categorical)

  dlrm_in_keras = DlrmInKeras(args)
  predict_logits = dlrm_in_keras(inputs, training=False)
  outputs = {'score': predict_logits}
  return tf.saved_model.predict_signature_def(inputs, outputs)


def main(args):
  dlrm_in_keras = DlrmInKeras(args)
  step = tf.train.get_or_create_global_step()
  lr = lr_with_linear_warmup_and_polynomial_decay(
    step,
    initial_value=args.lr_initial_value,
    warmup_steps=args.lr_warmup_steps,
    decay_start_step=args.lr_decay_start_step,
    decay_steps=args.lr_decay_steps)
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

  def loss_func(y_true, y_pred):
    return tf.reduce_mean(
      tf.keras.losses.binary_crossentropy(y_true, y_pred))

  if args.weights_dir is not None:
    dlrm_in_keras.load_weights(args.weights_dir)

  dlrm_in_keras.compile(
    loss=loss_func,
    metrics=[tf.keras.metrics.AUC()],
    optimizer=opt)

  if len(args.filenames) > 1:
    train_filenames = args.filenames[:-1]
    eval_filenames = args.filenames[-1]
  else:
    train_filenames = args.filenames
    eval_filenames = args.filenames

  train_dataset = dlrm_in_keras.input_dataset(
    train_filenames, args.train_batch_size)
  val_dataset = dlrm_in_keras.input_dataset(
    eval_filenames, args.eval_batch_size)

  if args.evaluate:
    dlrm_in_keras.evaluate(
      x=val_dataset,
      y=None,
      batch_size=args.eval_batch_size,
      checkpoint_dir=args.output_dir,
      steps=args.eval_max_steps)
  else:
    dlrm_in_keras.fit(
      x=train_dataset,
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

    dlrm_in_keras.summary()

    dlrm_in_keras.export_saved_model(
      args.output_dir,
      lambda: predict_fn(args))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--disable-imputation', default=False, action='store_true')
  parser.add_argument(
    '--use-ev', default=False, action='store_true')
  parser.add_argument(
    '--use-shared-emb-feats', type=str, default=None)
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
    '--bottom-mlp-dims', type=int, nargs='+', default=[512, 256])
  parser.add_argument(
    '--top-mlp-dims', type=int, nargs='+', default=[1024, 1024, 512, 256, 1])
  parser.add_argument(
    '--mlp-dims', type=int, nargs='+', default=[1024, 1024, 512, 256, 1])
  parser.add_argument('--train-batch-size', type=int, default=64000)
  parser.add_argument('--train-max-steps', type=int, default=None)
  parser.add_argument('--evaluate', default=False, action='store_true')
  parser.add_argument('--eval-batch-size', type=int, default=100)
  parser.add_argument('--eval-max-steps', type=int, default=1)
  parser.add_argument('--output-dir', default='./outputs')
  parser.add_argument('--load-ckpt-dir', default=None)
  parser.add_argument('--weights-dir', default=None)
  parser.add_argument(
    '--data-spec-filename', default='ranking/criteo/data/spec.json')
  parser.add_argument('filenames', nargs='+')
  parsed = parser.parse_args()
  parsed.data_spec = DataSpec.read(
    parsed.data_spec_filename,
    disable_imputation=parsed.disable_imputation,
    disable_transform=True,
    override_embedding_size=parsed.embedding_dim)
  with hb.scope():
    main(parsed)

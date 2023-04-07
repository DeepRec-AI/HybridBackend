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

r'''Training and evaluating DIN model over taobao dataset using Keras API.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from ranking.data import DataSpec
from ranking.model import din
import tensorflow as tf
from tensorflow.python.util import module_wrapper as _deprecation  # pylint: disable=protected-access

import hybridbackend.tensorflow as hb

_deprecation._PER_MODULE_WARNING_LIMIT = 0  # pylint: disable=protected-access
tf.get_logger().propagate = False


class DinModel(tf.keras.layers.Layer):
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
    self._categorical_dtypes = {
      fs.name: fs.dtype
      for fs in self._args.data_spec.feature_specs
      if fs.name not in ('ts', 'label') and fs.embedding is not None}
    self._varlen_categorical_fields = [
      fs.name for fs in self._args.data_spec.feature_specs
      if fs.name not in ('ts', 'label') and fs.type == 'list']
    self._query_fields = ['item_category', 'item_brand']
    self._history_fields = ['user_pv_category_list', 'user_pv_brand_list']
    self._sp_varlen_fields = [
      v for v in self._varlen_categorical_fields
      if v not in self._history_fields]
    self._embedding_weights = {}
    with hb.embedding_scope():
      with tf.device(self._args.embedding_weight_device):
        for f in self._categorical_fields:
          if self._args.use_ev:
            with tf.variable_scope(
                f'{f}_embedding',
                partitioner=tf.fixed_size_partitioner(
                  hb.context.world_size)):
              self._embedding_weights[f] = tf.get_embedding_variable(
                f'{f}_weight',
                key_dtype=self._categorical_dtypes[f],
                embedding_dim=self._args.data_spec.embedding_dims[f],
                initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
          else:
            self._embedding_weights[f] = tf.get_variable(
              f'{f}_weight',
              shape=(
                self._args.data_spec.embedding_sizes[f],
                self._args.data_spec.embedding_dims[f]),
              initializer=tf.random_uniform_initializer(-1e-3, 1e-3))

  def call(self, features, labels=None, **kwargs):
    r'''Model function for estimator.
    '''
    del kwargs

    for f in self._history_fields:
      if (not isinstance(features[f], tf.sparse.SparseTensor)
          and features[f].op.type != 'Placeholder'):
        raise ValueError('features value with a varying lenght shall be '
                         ' a tf.sparse.SparseTensor')
      if features[f].op.type != 'Placeholder':
        features[f] = tf.sparse.slice(
          features[f], [0, 0],
          [self._args.train_batch_size, self._args.max_varlength])

    with tf.device(self._args.transform_device):
      dense_value_list = [
        self._args.data_spec.transform_numeric(f, features[f])
        for f in self._numeric_fields]
      sparse_value_dict = {}
      for f in self._categorical_fields:
        if f in self._history_fields:
          sparse_value_dict[f] =\
            self._args.data_spec.transform_categorical_non_pooling(
              f, features[f], self._embedding_weights[f])
        else:
          sparse_value_dict[f] = self._args.data_spec.transform_categorical(
            f, features[f], self._embedding_weights[f])
        if f in self._history_fields:
          sparse_value_dict[f] = tf.reshape(
            sparse_value_dict[f],
            shape=[
              self._args.train_batch_size, self._args.max_varlength,
              self._args.data_spec.embedding_dims[f]])
        elif f in self._query_fields:
          sparse_value_dict[f] = tf.reshape(
            sparse_value_dict[f],
            shape=[
              self._args.train_batch_size, 1,
              self._args.data_spec.embedding_dims[f]])

    query_emb_list = [sparse_value_dict[f] for f in self._query_fields]
    key_emb_list = [sparse_value_dict[f] for f in self._history_fields]
    key_len = tf.constant(
      self._args.max_varlength, shape=[self._args.train_batch_size, 1],
      dtype=tf.int32)
    sequence_pooled_embed_list = [
      sparse_value_dict[f] for f in self._sp_varlen_fields]
    dnn_input_emb_list = [
      sparse_value_dict[f]
      for f in self._categorical_fields
      if f not in self._query_fields and f not in self._history_fields]
    dnn_input_emb_list += sequence_pooled_embed_list

    logits = din(
      dense_value_list,
      query_emb_list,
      key_emb_list,
      key_len,
      dnn_input_emb_list)
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

  predict_logits = DinModel(args)(inputs)
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

  model_output = DinModel(args)(features)
  din_in_keras = tf.keras.Model(inputs=[features], outputs=model_output)

  def loss_func(y_true, y_pred):
    return tf.reduce_mean(
      tf.keras.losses.binary_crossentropy(y_true, y_pred))

  opt = tf.train.AdagradOptimizer(learning_rate=args.lr)

  if args.weights_dir is not None:
    din_in_keras.load_weights(args.weights_dir)

  din_in_keras.compile(
    loss=loss_func,
    metrics=['binary_crossentropy'],
    optimizer=opt,
    target_tensors=labels)

  din_in_keras.fit(
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

  din_in_keras.export_saved_model(
    args.output_dir,
    lambda: predict_fn(args))


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
  parser.add_argument('--max-varlength', type=int, default=4)
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

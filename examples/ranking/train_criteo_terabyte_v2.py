#!/usr/bin/env python

r'''Training model for criteo terabyte dataset using Estimator API.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import hybridbackend.tensorflow as hb

from layers import Ranking
from optimization import sgd_decay_optimize


class CriteoTerabyteModel:
  r'''Model for criteo terabyte dataset.
  '''
  def __init__(self, args):
    r'''Creates a model.
    '''
    self.args = args
    self.label_field = 'label'
    self.numeric_fields = [f'f{i}' for i in range(13)]
    self.categorical_fields = [f'id{i}' for i in range(26)]
    self.embedding_bucket_sizes = [
      39884406,
      39043,
      17289,
      7420,
      20263,
      3,
      7120,
      1543,
      63,
      38532951,
      2953546,
      403346,
      10,
      2208,
      11938,
      155,
      4,
      976,
      14,
      39979771,
      25641295,
      39664984,
      585935,
      12972,
      108,
      36
    ]
    if args.max_bucket_size > 0:
      self.embedding_bucket_sizes = [
        n if n < args.max_bucket_size else args.max_bucket_size
        for n in self.embedding_bucket_sizes]

  def make_input(self, input_files, batch_size):
    r'''Make dataset for training and evaluation.
    '''
    with tf.device('/cpu:0'):
      ds = hb.data.ParquetDataset(
        input_files,
        batch_size=batch_size,
        num_parallel_reads=len(input_files),
        drop_remainder=True)
      ds = ds.apply(hb.data.to_sparse())

      def map_fn(batch):
        features = {}
        features.update({
          f: tf.to_float(batch[f]) for f in self.numeric_fields})
        features.update({
          f: batch[f] for f in self.categorical_fields})
        labels = tf.to_float(batch[self.label_field])
        return features, labels

      ds = ds.map(map_fn)
      ds = ds.prefetch(1)
    return ds

  def make_input_receiver(self):
    r'''Make placeholders for prediction.
    '''
    with tf.device('/cpu:0'):
      example_spec = {}
      for f in self.numeric_fields:
        example_spec[f] = tf.io.FixedLenFeature([1], dtype=tf.float32)
      for f in self.categorical_fields:
        example_spec[f] = tf.io.VarLenFeature(dtype=tf.int64)

      serialized_examples = tf.placeholder(
        dtype=tf.string,
        shape=[None],
        name='input')
      features = tf.io.parse_example(serialized_examples, example_spec)

    return tf.estimator.export.ServingInputReceiver(
      features, {'examples': serialized_examples})

  @hb.function(emb_device='/cpu:0')
  def call(self, features, labels, mode, params):
    r'''Train and evaluate.
    '''
    del params
    embedding_columns = [
      tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
          key=f,
          num_buckets=self.embedding_bucket_sizes[fid],
          default_value=0),
        dimension=self.args.dimension,
        initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
      for fid, f in enumerate(self.categorical_fields)]
    embeddings = hb.dense_features(
      {f: features[f] for f in self.categorical_fields},
      embedding_columns)
    values = tf.concat(
      [tf.reshape(features[f], [-1, 1]) for f in self.numeric_fields],
      axis=1)
    logits = Ranking(embedding_columns)(values, embeddings)

    if mode == tf.estimator.ModeKeys.TRAIN:
      labels = tf.reshape(labels, shape=[-1, 1])
      loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(labels, logits))
      train_op = sgd_decay_optimize(
        loss,
        lr_initial_value=self.args.lr_initial_value,
        lr_warmup_steps=self.args.lr_warmup_steps,
        lr_decay_start_step=self.args.lr_decay_start_step,
        lr_decay_steps=self.args.lr_decay_steps)

      _, update_auc = hb.metrics.auc(labels, logits, name='train_auc')
      train_op = tf.group([train_op, update_auc])

      chief_hooks = []
      if self.args.save_timeline_every_n_step:
        chief_hooks.append(
          tf.train.ProfilerHook(
            save_steps=self.args.save_timeline_every_n_step,
            output_dir=self.args.output_dir))

      return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_chief_hooks=chief_hooks)

    if mode == tf.estimator.ModeKeys.EVAL:
      labels = tf.reshape(labels, shape=[-1, 1])
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
  model = CriteoTerabyteModel(args)

  train_files = list(args.input_files)
  if len(train_files) > 1:
    eval_files = [train_files.pop()]
  else:
    eval_files = None
    args.eval_every_n_iter = None

  estimator = hb.estimator.Estimator(
    model.call,
    model_dir=args.output_dir,
    config=tf.estimator.RunConfig(
      save_checkpoints_steps=args.save_checkpoint_every_n_step,
      keep_checkpoint_max=2))

  estimator.train_and_evaluate(
    tf.estimator.TrainSpec(
      input_fn=lambda: model.make_input(
        train_files, args.train_batch_size),
      max_steps=args.train_max_steps),
    tf.estimator.EvalSpec(
      input_fn=lambda: model.make_input(
        eval_files, args.eval_batch_size),
      steps=args.eval_max_steps),
    eval_every_n_iter=args.eval_every_n_step)

  estimator.export_saved_model(
    args.savedmodel_dir,
    model.make_input_receiver)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--head', default='DLRM')
  parser.add_argument('--max-bucket-size', type=int, default=1000000)
  parser.add_argument('--dimension', type=int, default=64)
  parser.add_argument('--lr-initial-value', type=float, default=24.)
  parser.add_argument('--lr-warmup-steps', type=int, default=2750)
  parser.add_argument('--lr-decay-start-step', type=int, default=48000)
  parser.add_argument('--lr-decay-steps', type=int, default=27772)
  parser.add_argument('--output-dir', default='./outputs')
  parser.add_argument('--savedmodel-dir', default='./outputs/savedmodels')
  parser.add_argument('--save-checkpoint-every-n-step', type=int, default=1000)
  parser.add_argument('--save-timeline-every-n-step', type=int, default=None)
  parser.add_argument('--train-batch-size', type=int, default=64000)
  parser.add_argument('--train-max-steps', type=int, default=1000)
  parser.add_argument('--eval-batch-size', type=int, default=1000)
  parser.add_argument('--eval-every-n-step', type=int, default=100)
  parser.add_argument('--eval-max-steps', type=int, default=1)
  parser.add_argument('input_files', nargs='+')
  main(parser.parse_args())

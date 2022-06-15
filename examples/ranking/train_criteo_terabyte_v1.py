#!/usr/bin/env python

r'''Training model for criteo terabyte dataset using MonitoredSession API.
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

  def compute_logits(self, features):
    r'''Compute logits.
    '''
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
    return Ranking(embedding_columns)(values, embeddings)

  @hb.function(emb_device='/cpu:0')
  def train(self, input_files):
    r'''Train function.
    '''
    ds = self.make_input(input_files, self.args.train_batch_size)
    features, labels = hb.data.make_one_shot_iterator(ds).get_next()
    logits = self.compute_logits(features)
    labels = tf.reshape(labels, shape=[-1, 1])
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
    train_op = sgd_decay_optimize(
      loss,
      lr_initial_value=self.args.lr_initial_value,
      lr_warmup_steps=self.args.lr_warmup_steps,
      lr_decay_start_step=self.args.lr_decay_start_step,
      lr_decay_steps=self.args.lr_decay_steps)

    _, update_auc = hb.metrics.auc(labels, logits, name='train_auc')
    return tf.group([train_op, update_auc])

  @hb.function(emb_device='/cpu:0')
  def evaluate(self, input_files):
    r'''Evaluate function.
    '''
    ds = self.make_input(input_files, self.args.eval_batch_size)
    features, labels = hb.data.make_one_shot_iterator(ds).get_next()
    labels = tf.reshape(labels, shape=[-1, 1])
    logits = self.compute_logits(features)
    loss = tf.reduce_mean(
      tf.keras.losses.binary_crossentropy(labels, logits))

    return {
      'auc': hb.metrics.auc(
        labels=labels,
        predictions=logits,
        name='eval_auc'),
      'loss': (tf.no_op(), loss)}

  @hb.function(emb_device='/cpu:0')
  def predict(self):
    r'''Predict function.
    '''
    features = {}
    features.update({
      f: tf.placeholder(dtype=tf.float32)
      for f in self.numeric_fields})
    features.update({
      f: tf.placeholder(dtype=tf.int32, shape=[None])
      for f in self.categorical_fields})
    logits = self.compute_logits(features)

    return tf.saved_model.predict_signature_def(features, {'score': logits})


def main(args):
  model = CriteoTerabyteModel(args)

  train_files = list(args.input_files)
  if len(train_files) > 1:
    eval_files = [train_files.pop()]
  else:
    eval_files = None
    args.eval_every_n_iter = None

  with tf.Graph().as_default():
    train_op = model.train(train_files)

    hooks = [
      tf.train.LoggingTensorHook(
        {'global_step': tf.train.get_global_step()},
        every_n_iter=args.log_every_n_step)]
    if args.train_max_steps is not None:
      hooks.append(tf.estimator.StopAtStepHook(args.train_max_steps))

    chief_hooks = []
    if args.save_timeline_every_n_step:
      chief_hooks.append(
        tf.train.ProfilerHook(
          save_steps=args.save_timeline_every_n_step,
          output_dir=args.output_dir))

    with hb.train.monitored_session(
        hooks=hooks,
        chief_only_hooks=chief_hooks,
        checkpoint_dir=args.output_dir,
        save_checkpoint_steps=args.save_checkpoint_every_n_step,
        eval_every_n_iter=args.eval_every_n_step,
        eval_fn=lambda: model.evaluate(eval_files)) as sess:
      while not sess.should_stop():
        sess.run(train_op)

  hb.saved_model.export(
    args.savedmodel_dir,
    tf.train.latest_checkpoint(args.output_dir),
    model.predict)


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
  parser.add_argument('--log-every-n-step', type=int, default=100)
  parser.add_argument('--save-checkpoint-every-n-step', type=int, default=1000)
  parser.add_argument('--save-timeline-every-n-step', type=int, default=None)
  parser.add_argument('--train-batch-size', type=int, default=64000)
  parser.add_argument('--train-max-steps', type=int, default=1000)
  parser.add_argument('--eval-batch-size', type=int, default=1000)
  parser.add_argument('--eval-every-n-step', type=int, default=100)
  parser.add_argument('--eval-max-steps', type=int, default=1)
  parser.add_argument('input_files', nargs='+')
  main(parser.parse_args())

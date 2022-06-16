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

r'''Utilities of estimators using hybrid backends.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import threading

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import estimator_training
from tensorflow.python.eager import context as _context
from tensorflow.python.estimator import estimator as _estimator
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.model_utils import mode_keys
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
from tensorflow_estimator.python.estimator import estimator as _estimator_lib
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
from tensorflow_estimator.python.estimator.training import _TrainingExecutor

from hybridbackend.tensorflow.data.iterators import make_one_shot_iterator
from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.context import context_scope
from hybridbackend.tensorflow.framework.device import device_function
from hybridbackend.tensorflow.saved_model.simple_save import export_all
from hybridbackend.tensorflow.training.eval import eval_scope
from hybridbackend.tensorflow.training.eval import EvaluationHook
from hybridbackend.tensorflow.training.function import configure
from hybridbackend.tensorflow.training.function import scope
from hybridbackend.tensorflow.training.saver import \
  HybridBackendSaverBuilderBase
from hybridbackend.tensorflow.training.saver import Saver
from hybridbackend.tensorflow.training.server import wraps_server


class RunConfig(run_config.RunConfig):
  r'''RunConfig for estimators.
  '''
  @classmethod
  def build(cls, prototype=None, **kwargs):
    r'''Creates RunConfig from prototype.
    '''
    if prototype is None:
      return cls(**kwargs)
    prototype = prototype.replace(device_fn=device_function)
    prototype._is_chief = True  # pylint: disable=protected-access
    prototype._session_config = configure(prototype=prototype.session_config)  # pylint: disable=protected-access
    if prototype._evaluation_master == '':  # pylint: disable=protected-access
      prototype._evaluation_master = prototype.master  # pylint: disable=protected-access

    return prototype

  def __init__(self, **kwargs):
    r'''Creates a wrapped RunConfig.
    '''
    kwargs['session_config'] = configure(
      prototype=kwargs.pop('session_config', None))
    kwargs['device_fn'] = device_function
    super().__init__(**kwargs)
    self._is_chief = True  # pylint: disable=protected-access


def wraps_model_fn(model_fn, model_dir, config):
  r'''Decorator to set params in a model function.
  '''
  def wrapped_model_fn(features, labels, mode, params):
    r'''Wrapped model function.
    '''
    with scope(mode=mode, model_dir=model_dir):
      estimator_spec = model_fn(features, labels, mode, params)
    if estimator_spec.scaffold.saver:
      if not isinstance(
          estimator_spec.scaffold.saver._builder,  # pylint: disable=protected-access
          HybridBackendSaverBuilderBase):
        raise ValueError(
          'scaffold.saver in EstimatorSpec must be hb.train.Saver, '
          'you can try call hb.train.replace_default_saver() before '
          'creation of the scaffold.')
    else:
      estimator_spec.scaffold._saver = Saver(  # pylint: disable=protected-access
        max_to_keep=config.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=config.keep_checkpoint_every_n_hours,
        defer_build=True,
        save_relative_paths=True)
    training_hooks = list(estimator_spec.training_hooks) or []
    training_hooks += Context.get().training_hooks
    training_chief_hooks = list(estimator_spec.training_chief_hooks) or []
    training_chief_hooks += Context.get().training_chief_hooks
    estimator_spec = estimator_spec._replace(  # pylint: disable=protected-access
      training_hooks=training_hooks,
      training_chief_hooks=training_chief_hooks)
    return estimator_spec
  return wrapped_model_fn


class PatchTensorflowAPIForEstimator(object):  # pylint: disable=useless-object-inheritance
  r'''Context manager that patches TF APIs for estimator.
  '''
  _lock = threading.Lock()
  _stack_depth = 0

  def __init__(self, drop_remainder):
    self._drop_remainder = drop_remainder

  def __enter__(self):
    with PatchTensorflowAPIForEstimator._lock:
      PatchTensorflowAPIForEstimator._stack_depth += 1
      if PatchTensorflowAPIForEstimator._stack_depth <= 1:
        def wraps_parse_input_fn_result(parse_fn):  # pylint: disable=unused-argument
          r'''replaces iterator.
          '''
          def wrapped_parse_input_fn_result(result):
            r'''Wrapped parse_input_fn_result.
            '''
            input_hooks = []
            if isinstance(result, (dataset_ops.Dataset, dataset_ops.DatasetV2)):
              iterator = make_one_shot_iterator(result, self._drop_remainder)
              result = iterator.get_next()
            return estimator_util.parse_iterator_result(result) + (input_hooks,)
          return wrapped_parse_input_fn_result
        self._prev_parse_input_fn_result = estimator_util.parse_input_fn_result
        estimator_util.parse_input_fn_result = wraps_parse_input_fn_result(
          self._prev_parse_input_fn_result)

      return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    with PatchTensorflowAPIForEstimator._lock:
      if PatchTensorflowAPIForEstimator._stack_depth <= 1:
        estimator_util.parse_input_fn_result = self._prev_parse_input_fn_result
      PatchTensorflowAPIForEstimator._stack_depth -= 1


def wraps_estimator(cls):
  r'''Estimator decorator to train and evaluate in parallel.
  '''
  class HybridBackendEstimator(cls):
    r'''Class to train and evaluate TensorFlow models.
    '''
    def __init__(self, model_fn, **kwargs):
      r'''Constructs a wrapped `Estimator` instance.

      Args:
        model_fn: Model function. See
          `tensorflow_estimator/python/estimator/estimator.py#L145`
          for more information.
        kwargs: Estimator arguments.
      '''
      kwargs['config'] = RunConfig.build(prototype=kwargs.pop('config', None))
      model_dir = kwargs.get('model_dir', None)
      self._train_drop_remainder = kwargs.pop('train_drop_remainder', None)
      self._eval_drop_remainder = kwargs.pop('eval_drop_remainder', None)
      self._predict_drop_remainder = kwargs.pop('predict_drop_remainder', None)

      super().__init__(
        wraps_model_fn(model_fn, model_dir, kwargs['config']),
        **kwargs)

    def _assert_members_are_not_overridden(self):
      r'''disable the overridden check here.
      '''

    def train(
        self, input_fn, hooks=None, max_steps=None, saving_listeners=None):
      r'''support detect_end_dataset in training.
      '''
      if saving_listeners is None:
        saving_listeners = []
      saving_listeners.extend(Context.get().saving_listeners)
      with context_scope(
          mode=mode_keys.EstimatorModeKeys.TRAIN,
          model_dir=self._model_dir):
        with PatchTensorflowAPIForEstimator(self._train_drop_remainder):
          return super().train(
            input_fn, hooks=hooks, max_steps=max_steps,
            saving_listeners=saving_listeners)

    def _actual_eval(
        self, input_fn, strategy=None, steps=None, hooks=None,
        checkpoint_path=None, name=None):
      r'''standalone evaluation methods supports HB
      '''
      if strategy:
        raise ValueError('DistributionStrategy not supported')

      with _context.graph_mode(), context_scope(
          comm_pool_capacity=1,
          comm_pool_name=mode_keys.EstimatorModeKeys.EVAL,
          mode=mode_keys.EstimatorModeKeys.EVAL,
          model_dir=self._model_dir):
        hooks = _estimator._check_hooks_type(hooks)  # pylint: disable=protected-access
        hooks.extend(self._convert_eval_steps_to_hooks(steps))  # pylint: disable=protected-access
        if not checkpoint_path:
          latest_path = checkpoint_management.latest_checkpoint(self._model_dir)  # pylint: disable=protected-access
          if not latest_path:
            raise ValueError(
              f'Could not find trained model in model_dir: {self._model_dir}.')  # pylint: disable=protected-access
          checkpoint_path = latest_path

        with ops.Graph().as_default() as g, g.device(self._device_fn):  # pylint: disable=protected-access
          with eval_scope(), PatchTensorflowAPIForEstimator(
              self._eval_drop_remainder):
            (scaffold, update_op, eval_dict, all_hooks) = (
              self._evaluate_build_graph(  # pylint: disable=protected-access
                input_fn,
                hooks, checkpoint_path))
            return self._evaluate_run(  # pylint: disable=protected-access
              checkpoint_path=checkpoint_path,
              scaffold=scaffold,
              update_op=update_op,
              eval_dict=eval_dict,
              all_hooks=all_hooks,
              output_dir=self.eval_dir(name))

    def train_and_evaluate(
        self, train_spec, eval_spec,
        eval_every_n_iter=None):
      r'''Train and evaluate the `estimator`.

      Args:
        eval_every_n_iter: `int`, runs parallel evaluation once every
          N training iteration. If None, disable the evaluation
      '''
      ctx = Context.get()
      if eval_every_n_iter is not None:
        def _eval_fn():
          with PatchTensorflowAPIForEstimator(self._eval_drop_remainder):
            with context_scope(model_dir=self._model_dir):
              (_, evaluation_hooks, input_hooks, update_op, metrics) = (
                self._call_model_fn_eval(  # pylint: disable=protected-access
                  eval_spec.input_fn, self.config))
              hooks = list(evaluation_hooks) or []
              hooks.extend(list(input_hooks) or [])
              return update_op, metrics, hooks
        eval_summary_dir = self.eval_dir(
          name=f'{ctx.rank}' if ctx.world_size > 1 else '')
        ctx.add_training_hook(
          EvaluationHook(
            _eval_fn,
            steps=eval_spec.steps,
            every_n_iter=eval_every_n_iter,
            summary_dir=eval_summary_dir))

      if self.config.cluster_spec:
        executor = TrainingExecutor(
          estimator=self,
          train_spec=train_spec,
          eval_spec=eval_spec)
        if estimator_training.should_run_distribute_coordinator(self.config):
          raise ValueError(
            'Running `train_and_evaluate` with Distribute Coordinator '
            'not supported.')
        return executor.run()

      return self.train(
        train_spec.input_fn,
        hooks=train_spec.hooks,
        max_steps=train_spec.max_steps)

    def export_saved_model(
        self, export_dir_base, serving_input_receiver_fn,
        assets_extra=None,
        as_text=False,
        checkpoint_path=None,
        experimental_mode=ModeKeys.PREDICT,
        **kwargs):
      r'''Exports inference graph as a `SavedModel` into the given dir.
      '''
      if not serving_input_receiver_fn:
        raise ValueError('An input_receiver_fn must be defined.')

      input_receiver_fn_map = {experimental_mode: serving_input_receiver_fn}

      return self._export_all_saved_models(
        export_dir_base,
        input_receiver_fn_map,
        assets_extra=assets_extra,
        as_text=as_text,
        checkpoint_path=checkpoint_path,
        strip_default_attrs=True,
        **kwargs)

    def experimental_export_all_saved_models(
        self, export_dir_base, input_receiver_fn_map,
        assets_extra=None,
        as_text=False,
        checkpoint_path=None,
        **kwargs):
      r'''Exports a `SavedModel` with `tf.MetaGraphDefs` for each requested
        mode.
      '''
      return self._export_all_saved_models(
        export_dir_base, input_receiver_fn_map,
        assets_extra=assets_extra,
        as_text=as_text,
        checkpoint_path=checkpoint_path,
        strip_default_attrs=True,
        **kwargs)

    def _export_all_saved_models(
        self,
        export_dir_base,
        input_receiver_fn_map,
        assets_extra=None,
        as_text=False,
        checkpoint_path=None,
        strip_default_attrs=True,
        **kwargs):
      r'''Exports multiple modes in the model function to a SavedModel.
      '''
      if (input_receiver_fn_map.get(ModeKeys.TRAIN)
          or input_receiver_fn_map.get(ModeKeys.EVAL)
          or not input_receiver_fn_map.get(ModeKeys.PREDICT)):
        raise ValueError('Only PREDICT mode is supported.')
      mode = ModeKeys.PREDICT

      if Context.get().rank != 0:
        return None

      if not checkpoint_path:
        checkpoint_path = checkpoint_management.latest_checkpoint(
          self._model_dir)
      if not checkpoint_path:
        if self._warm_start_settings:
          checkpoint_path = self._warm_start_settings.ckpt_to_initialize_from
          if gfile.IsDirectory(checkpoint_path):
            checkpoint_path = checkpoint_management.latest_checkpoint(
              checkpoint_path)
        else:
          raise ValueError(
            f'Couldn\'t find trained model at {self._model_dir}.')

      def _fn():
        random_seed.set_random_seed(self._config.tf_random_seed)

        input_receiver_fn = input_receiver_fn_map[mode]
        input_receiver = input_receiver_fn()
        estimator_spec = self._call_model_fn(
          features=input_receiver.features,
          labels=getattr(input_receiver, 'labels', None),
          mode=mode,
          config=self.config)
        export_outputs = export_lib.export_outputs_for_mode(
          mode=estimator_spec.mode,
          serving_export_outputs=estimator_spec.export_outputs,
          predictions=estimator_spec.predictions,
          loss=estimator_spec.loss,
          metrics=estimator_spec.eval_metric_ops)
        signature_def_map = export_lib.build_all_signature_defs(
          input_receiver.receiver_tensors,
          export_outputs,
          getattr(input_receiver, 'receiver_tensors_alternatives', None),
          serving_only=(mode == ModeKeys.PREDICT))
        main_op = None
        if estimator_spec.scaffold.local_init_op is not None:
          main_op = estimator_spec.scaffold.local_init_op
        return signature_def_map, main_op

      return export_all(
        export_dir_base,
        checkpoint_path,
        _fn,
        assets_extra=assets_extra,
        as_text=as_text,
        clear_devices=True,
        strip_default_attrs=strip_default_attrs,
        modes=[mode],
        **kwargs)

    def predict(
        self, input_fn,
        predict_keys=None,
        hooks=None,
        checkpoint_path=None,
        yield_single_examples=True):
      r'''Predict method of estimator in HB.
      '''
      _estimator_lib._estimator_api_gauge.get_cell('predict').set(True)  # pylint: disable=protected-access
      with _context.graph_mode(), context_scope(
          mode=mode_keys.EstimatorModeKeys.PREDICT,
          model_dir=self._model_dir,
          comm_pool_capacity=1,
          comm_pool_name=mode_keys.EstimatorModeKeys.PREDICT):
        hooks = _estimator_lib._check_hooks_type(hooks)  # pylint: disable=protected-access
        # Check that model has been trained.
        if not checkpoint_path:
          checkpoint_path = checkpoint_management.latest_checkpoint(
            self._model_dir)
        if not checkpoint_path:
          logging.info(
            f'Could not find trained model in model_dir: {self._model_dir},'
            f'running initialization to predict.')
        with ops.Graph().as_default() as g, g.device(self._device_fn):
          with ops.name_scope(mode_keys.EstimatorModeKeys.PREDICT):
            random_seed.set_random_seed(self._config.tf_random_seed)
            self._create_and_assert_global_step(g)
            features, input_hooks = self._get_features_from_input_fn(
              input_fn, ModeKeys.PREDICT)
            estimator_spec = self._call_model_fn(
              features, None, ModeKeys.PREDICT, self.config)

          # Call to warm_start has to be after model_fn is called.
          self._maybe_warm_start(checkpoint_path)

          predictions = self._extract_keys(estimator_spec.predictions,
                                           predict_keys)
          all_hooks = list(input_hooks)
          all_hooks.extend(hooks)
          all_hooks.extend(list(estimator_spec.prediction_hooks or []))
          with monitored_session.MonitoredSession(
              session_creator=monitored_session.ChiefSessionCreator(
                checkpoint_filename_with_path=checkpoint_path,
                master=self._config.master,
                scaffold=estimator_spec.scaffold,
                config=self._session_config),
              hooks=all_hooks) as mon_sess:
            while not mon_sess.should_stop():
              preds_evaluated = mon_sess.run(predictions)
              if not yield_single_examples:
                yield preds_evaluated
              elif not isinstance(predictions, dict):
                for pred in preds_evaluated:
                  yield pred
              else:
                for i in range(self._extract_batch_length(preds_evaluated)):
                  yield {
                    key: value[i]
                    for key, value in six.iteritems(preds_evaluated)
                  }

  return HybridBackendEstimator


Estimator = wraps_estimator(_estimator.Estimator)


class TrainingExecutor(_TrainingExecutor):
  r'''The executor to run `Estimator` training and evaluation.
  '''
  def _start_std_server(self, config):
    r'''Creates, starts, and returns a server_lib.Server.'''
    logging.info('Start Tensorflow server.')
    return wraps_server(server_lib.Server)(
      config.cluster_spec,
      job_name=config.task_type,
      task_index=config.task_id,
      config=configure(prototype=config.session_config),
      start=True,
      protocol=config.protocol)


def train_and_evaluate(
    estimator, train_spec, eval_spec,
    eval_every_n_iter=None):
  r'''Train and evaluate the `estimator`.

  Args:
    eval_every_n_iter: `int`, runs parallel evaluation once every
      N training iteration. If None, disable the evaluation
  '''
  if not isinstance(estimator, Estimator):
    raise TypeError('estimator must be `hb.estimator.Estimator`')

  return estimator.train_and_evaluate(
    train_spec, eval_spec,
    eval_every_n_iter=eval_every_n_iter)

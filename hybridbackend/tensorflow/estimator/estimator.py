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

from tensorflow.python.distribute import estimator_training
from tensorflow.python.estimator import estimator as _estimator
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib

from tensorflow_estimator.python.estimator import model_fn as _model_fn
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
from tensorflow_estimator.python.estimator.training import _TrainingExecutor
from tensorflow_estimator.python.estimator.training import \
  TrainSpec as _TrainSpec
from tensorflow_estimator.python.estimator.training import \
  EvalSpec as _EvalSpec

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.device import device_function
from hybridbackend.tensorflow.saved_model.simple_save import export_all
from hybridbackend.tensorflow.training.function import function_extent
from hybridbackend.tensorflow.training.saver import \
  HybridBackendSaverBuilderBase
from hybridbackend.tensorflow.training.saver import Saver
from hybridbackend.tensorflow.training.server_lib import build_session_config
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
    prototype._session_config = build_session_config(prototype.session_config)  # pylint: disable=protected-access
    if prototype._evaluation_master == '':  # pylint: disable=protected-access
      prototype._evaluation_master = prototype.master  # pylint: disable=protected-access

    return prototype

  def __init__(self, **kwargs):
    r'''Creates a wrapped RunConfig.
    '''
    kwargs['session_config'] = build_session_config(
      kwargs.pop('session_config', None))
    kwargs['device_fn'] = device_function
    super().__init__(**kwargs)
    self._is_chief = True  # pylint: disable=protected-access


class EstimatorSpec(_model_fn.EstimatorSpec):
  r'''Wrapped EstimatorSpec.
  '''
  @classmethod
  def build(cls, prototype=None, **kwargs):
    r'''Creates EstimatorSpec from prototype.
    '''
    if prototype is None:
      return cls(**kwargs)

    return cls(
      prototype.mode,
      predictions=prototype.predictions,
      loss=prototype.loss,
      train_op=prototype.train_op,
      eval_metric_ops=prototype.eval_metric_ops,
      export_outputs=prototype.export_outputs,
      training_chief_hooks=prototype.training_chief_hooks,
      training_hooks=prototype.training_hooks,
      scaffold=prototype.scaffold,
      evaluation_hooks=prototype.evaluation_hooks,
      prediction_hooks=prototype.prediction_hooks,
      **kwargs)

  def __new__(
      cls,
      mode,
      predictions=None,
      loss=None,
      train_op=None,
      eval_metric_ops=None,
      export_outputs=None,
      training_chief_hooks=None,
      training_hooks=None,
      scaffold=None,
      evaluation_hooks=None,
      prediction_hooks=None,
      **kwargs):
    training_hooks = training_hooks or []
    training_hooks += Context.get().training_hooks
    if scaffold:
      scaffold_saver = scaffold.saver
      if scaffold_saver is None:
        scaffold._saver = Saver()  # pylint: disable=protected-access
      elif not isinstance(
          scaffold_saver._builder, HybridBackendSaverBuilderBase):
        raise ValueError(
          'scaffold.saver in EstimatorSpec must be hb.train.Saver, '
          'you can try call hb.train.replace_default_saver() before '
          'creation of the scaffold.')
    else:
      scaffold = monitored_session.Scaffold(saver=Saver())
    return super().__new__(
      cls, mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      export_outputs=export_outputs,
      training_chief_hooks=training_chief_hooks,
      training_hooks=training_hooks,
      scaffold=scaffold,
      evaluation_hooks=evaluation_hooks,
      prediction_hooks=prediction_hooks,
      **kwargs)


def wraps_model_fn(model_fn):
  r'''Decorator to set params in a model function.
  '''
  def wrapped_model_fn(features, labels, mode, params):
    r'''Wrapped model function.
    '''
    with function_extent():
      estimator_spec = model_fn(features, labels, mode, params)
    if not isinstance(estimator_spec, EstimatorSpec):
      raise ValueError('model_fn must return hb.estimator.EstimatorSpec')
    return estimator_spec
  return wrapped_model_fn


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
      super().__init__(wraps_model_fn(model_fn), **kwargs)

    def _assert_members_are_not_overridden(self):
      r'''disable the overridden check here
      '''

    def _export_all_saved_models(
        self,
        export_dir_base,
        input_receiver_fn_map,
        assets_extra=None,
        as_text=False,
        checkpoint_path=None,
        strip_default_attrs=True):
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
        modes=[mode])

  return HybridBackendEstimator


Estimator = wraps_estimator(_estimator.Estimator)


class TrainingExecutor(_TrainingExecutor):
  r'''The executor to run `Estimator` training and evaluation.
  '''
  def _start_distributed_training(self, saving_listeners=None):
    r'''Calls `Estimator` train in a distributed setting.
    '''
    if saving_listeners is None:
      saving_listeners = []
    super()._start_distributed_training(saving_listeners=saving_listeners)

  def _start_std_server(self, config):
    r'''Creates, starts, and returns a server_lib.Server.'''
    logging.info('Start Tensorflow server.')
    return wraps_server(server_lib.Server)(
      config.cluster_spec,
      job_name=config.task_type,
      task_index=config.task_id,
      config=build_session_config(config.session_config),
      start=True,
      protocol=config.protocol)


def train_and_evaluate(estimator, train_spec, eval_spec):
  r'''Train and evaluate the `estimator`.

  Args:
    eval_every_n_iter: `int`, runs parallel evaluation once every
      N training iteration. If None, disable the evaluation
  '''
  if not isinstance(estimator, Estimator):
    raise TypeError('estimator must be `hb.estimator.Estimator`')

  train_spec_dict = train_spec._asdict()
  eval_spec_dict = eval_spec._asdict()
  train_spec_actual = _TrainSpec(**train_spec_dict)
  eval_spec_actual = _EvalSpec(**eval_spec_dict)

  if estimator.config.cluster_spec:
    executor = TrainingExecutor(
      estimator=estimator,
      train_spec=train_spec_actual,
      eval_spec=eval_spec_actual)
    if estimator_training.should_run_distribute_coordinator(estimator.config):
      raise ValueError(
        'Running `train_and_evaluate` with Distribute Coordinator '
        'not supported.')
    return executor.run()

  return estimator.train(
    train_spec_actual.input_fn,
    hooks=train_spec_actual.hooks,
    max_steps=train_spec_actual.max_steps)

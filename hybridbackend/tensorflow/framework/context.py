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

r'''Context class for cluster and servers.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from six.moves import xrange  # pylint: disable=redefined-builtin
from six import string_types as string

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.framework import device as pydev
try:
  from tensorflow.python.training import device_util
except: # pylint: disable=bare-except
  from tensorflow.python.distribute import device_util
from tensorflow.python.training import server_lib


class Context(object): # pylint: disable=useless-object-inheritance
  r'''Configurations for cluster and servers.
  '''
  DEFAULT_DEVICE = '/job:localhost'

  _instance = None

  @classmethod
  def get(cls):
    r'''Get singleton.
    '''
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  @classmethod
  def current_device(cls):
    r'''Current device.
    '''
    return device_util.canonicalize(
        device_util.current(), default=cls.DEFAULT_DEVICE)

  @classmethod
  def canonicalize(cls, devices):
    r'''Canonicalize devices.
    '''
    return [
        device_util.canonicalize(d.strip(), default=cls.DEFAULT_DEVICE)
        for d in devices]

  @classmethod
  def set_tf_config(
      cls, task_type, task_id, worker_hosts, ps_hosts=None,
      has_evaluator=False):
    r'''Update TF_CONFIG environment variable.

    Args:
      task_type: name of current job. 'worker' should be set for
                 'chief' or 'evaluator'.
      task_id: index of current task.
      worker_hosts: List of workers.
      ps_hosts: (Optional.) List of parameter servers. Empty by default.
      has_evaluator: (Optional.) True if evaluator role is required.
                      False by default.
    '''
    tf_config = {}
    tf_config["task"] = {}
    tf_config["task"]["type"] = task_type
    tf_config["task"]["index"] = task_id
    tf_config["cluster"] = {}
    if worker_hosts:
      tf_config["cluster"]["chief"] = [worker_hosts[0]]
      if len(worker_hosts) > 1:
        if has_evaluator:
          tf_config["cluster"]["evaluator"] = [worker_hosts[1]]
          if len(worker_hosts) > 2:
            tf_config["cluster"]["worker"] = worker_hosts[2:]
          if task_type == "worker":
            if task_id == 0:
              tf_config["task"]["type"] = "chief"
              tf_config["task"]["index"] = 0
            elif task_id == 1:
              tf_config["task"]["type"] = "evaluator"
              tf_config["task"]["index"] = 0
            else:
              tf_config["task"]["index"] -= 2
        else:
          tf_config["cluster"]["worker"] = worker_hosts[1:]
          if task_type == "worker":
            if task_id == 0:
              tf_config["task"]["type"] = "chief"
              tf_config["task"]["index"] = 0
            else:
              tf_config["task"]["index"] -= 1
      else:
        if task_type == "worker":
          tf_config["task"]["type"] = "chief"
          tf_config["task"]["index"] = 0
    if ps_hosts:
      tf_config["cluster"]["ps"] = ps_hosts
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

  @classmethod
  def get_tf_config(cls):
    r'''Get configuration from TF_CONFIG environment variable.
    '''
    tf_config = json.loads(os.getenv('TF_CONFIG', '{}'))
    if not tf_config:
      return None
    task = tf_config['task']
    cluster = tf_config['cluster']
    task_type = task['type']
    task_id = int(task['index'])
    tf_config_type = collections.namedtuple(
        "TfConfig", ["task_type", "task_id", "cluster"])
    return tf_config_type(task_type, task_id, cluster)

  def __init__(self):
    r'''Construct a server specification.
    '''
    self._task_type = 'localhost'
    self._task_id = 0
    self._cluster_spec = None
    self._is_chief = True
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
      self._num_gpus = len(visible_devices.split(','))
    else:
      self._num_gpus = 1
    self._update()
    self._params = {}
    self._training_hooks = []
    self._training_chief_hooks = []
    self._evaluation_hooks = []
    self._prediction_hooks = []

  def __str__(self):
    return f'Context {self._task_type}:{self._task_id} ' + \
           f'{"chief" if self._is_chief else ""} {self._num_gpus}GPU, ' + \
           f'local={self._local_devices}, all={self._devices}'

  @property
  def cluster_spec(self):
    r'''cluster spec.
    '''
    return self._cluster_spec

  @property
  def task_type(self):
    r'''job name of current server. `localhost` by default.
    '''
    return self._task_type

  @property
  def task_id(self):
    r'''task index of current server. 0 by default.
    '''
    return self._task_id

  @property
  def target(self):
    r'''target of current server.
    '''
    if not self._cluster_spec:
      return ''

    addr = self._cluster_spec.job_tasks(self._task_type)[self._task_id]
    return f'grpc://{addr}'

  @property
  def is_chief(self):
    r'''True if current server is chief worker.
    '''
    return self._is_chief

  @property
  def has_gpu(self):
    r'''True if current server has GPU.
    '''
    return self._num_gpus > 0

  @property
  def num_gpus(self):
    r'''Number of GPUs.
    '''
    return self._num_gpus

  @property
  def default_device(self):
    r'''default device of current server.
    '''
    return self._default_device

  @property
  def local_devices(self):
    r'''devices of current server.
    '''
    return self._local_devices

  @property
  def devices(self):
    r'''devices of all servers.
    '''
    return self._devices

  @property
  def local_cpu_device(self):
    r'''CPU0 device of current server.
    '''
    return self._local_cpu_device

  @property
  def cpu_devices(self):
    r'''CPU devices of all servers.
    '''
    return self._cpu_devices

  @property
  def world_size(self):
    r'''Number of devices.
    '''
    return len(self._devices)

  @property
  def tower_index(self):
    r'''Local index of default local device.
    '''
    tower_index = self.param('tower_index', 0)
    if self.has_gpu:
      if tower_index >= len(self.local_devices):
        raise ValueError(
            f'Invalid tower_index {tower_index} (>= {len(self.local_devices)})')
    else:
      if tower_index > 0:
        raise ValueError('tower_index must not be set for CPU-only worker')
    return tower_index

  @property
  def rank(self):
    r'''Global index of default local device.
    '''
    return self.rank_at(self.tower_index)

  def rank_at(self, device_or_tower_id):
    r'''Get global index of device or tower_id.
    '''
    if isinstance(device_or_tower_id, string):
      return self._devices.index(device_or_tower_id)
    device_or_tower_id = int(device_or_tower_id)

    if self._num_gpus == 0:
      local_device = '/cpu:0'
      if device_or_tower_id > 0:
        raise ValueError('Only 1 tower allowed for CPU-only worker')
    else:
      if device_or_tower_id >= self._num_gpus:
        raise ValueError(
            f'Tower {device_or_tower_id} does not exist in the worker ' + \
            f'with {self._num_gpus} towers')
      local_device = f'/gpu:{device_or_tower_id}'
    local_device = pydev.DeviceSpec.from_string(
        f'/job:{self.task_type}/task:{self.task_id}{local_device}')
    local_device = device_util.canonicalize(
        local_device.to_string(),
        default=self.DEFAULT_DEVICE)
    return self._devices.index(local_device)

  def current_index(self):
    r'''Get global index of current device.
    '''
    return self._devices.index(Context.current_device())

  def _update(self, task_type=None, task_id=None, cluster_spec=None,
              num_gpus=None):
    r'''Update parameters from cluster_spec.

    If task_type, task_id or cluster_spec is None, these arguments will not be
    changed.

    Args:
      task_type: (Optional.) name of current job. `localhost` by default.
      task_id: (Optional.) index of current task. 0 by default.
      cluster_spec: (Optional.) ClusterSpec object.
    '''
    tf_config = None
    try:
      tf_config = self.get_tf_config()
    except: # pylint: disable=bare-except
      pass
    if tf_config:
      self._task_type = tf_config.task_type
      self._task_id = tf_config.task_id
      self._cluster_spec = server_lib.ClusterSpec(tf_config.cluster)
    else:
      self._task_type = 'localhost'
      self._task_id = 0
      self._cluster_spec = None
    if task_type:
      self._task_type = task_type
    if self._task_type not in ('localhost', 'chief', 'worker'):
      return

    if task_id:
      self._task_id = task_id
    if cluster_spec:
      self._cluster_spec = cluster_spec
    if self._cluster_spec:
      self._cluster_spec = multi_worker_util.normalize_cluster_spec(
          self._cluster_spec)
      self._is_chief = False
      try:
        self._is_chief = multi_worker_util.is_chief(
            self._cluster_spec, self._task_type, self._task_id)
      except: # pylint: disable=bare-except
        pass
    if num_gpus:
      self._num_gpus = num_gpus
    elif not self._num_gpus:
      num_gpus = 0
      num_gpus_config = config_pb2.ConfigProto()
      num_gpus_config.inter_op_parallelism_threads = 1
      num_gpus_config.intra_op_parallelism_threads = 1
      num_gpus_config.gpu_options.allow_growth = True
      for device in device_lib.list_local_devices(num_gpus_config):
        if device.device_type == 'GPU':
          num_gpus += 1
      self._num_gpus = num_gpus
    self._default_device = \
        f'/job:{self._task_type}/replica:0/task:{self._task_id}'
    self._local_cpu_device = \
        device_util.canonicalize('/device:CPU:0', default=self._default_device)
    if self._num_gpus == 0:
      self._local_devices = [self._local_cpu_device]
    else:
      self._local_devices = [
          device_util.canonicalize(
              f'/device:GPU:{d}', default=self._default_device) \
              for d in xrange(self._num_gpus)]
    if not self._cluster_spec:
      self._devices = list(self._local_devices)
      return
    task_indices = []
    try:
      task_defs = dict(enumerate(self._cluster_spec.job_tasks(self._task_type)))
      task_indices = sorted(task_defs, key=task_defs.__getitem__)
    except: # pylint: disable=bare-except
      pass
    worker_indices = []
    try:
      worker_defs = dict(enumerate(self._cluster_spec.job_tasks('worker')))
      worker_indices = sorted(worker_defs, key=worker_defs.__getitem__)
    except: # pylint: disable=bare-except
      pass
    chief_indices = []
    try:
      chief_defs = dict(enumerate(self._cluster_spec.job_tasks('chief')))
      chief_indices = sorted(chief_defs, key=chief_defs.__getitem__)
    except: # pylint: disable=bare-except
      pass
    self._cpu_devices = [
        device_util.resolve(
            f'/job:{self._task_type}/task:{t}/device:CPU:0') \
            for t in task_indices]
    if self._num_gpus == 0:
      self._devices = self._cpu_devices
      if self._task_type == 'worker':
        self._devices = [
            device_util.resolve(
                f'/job:chief/task:{t}/device:CPU:0') \
                for t in chief_indices] + self._devices
      elif self._task_type == 'chief':
        self._devices += [
            device_util.resolve(
                f'/job:worker/task:{t}/device:CPU:0') \
                for t in worker_indices]
      return
    self._devices = [
        device_util.resolve(
            f'/job:{self._task_type}/task:{t}/device:GPU:{g}') \
            for t in task_indices for g in xrange(self._num_gpus)]
    if self._task_type == 'worker':
      self._devices = [
          device_util.resolve(
              f'/job:chief/task:{t}/device:GPU:{g}') \
              for t in chief_indices for g in xrange(self._num_gpus)] + \
              self._devices
    elif self._task_type == 'chief':
      self._devices += [
          device_util.resolve(
              f'/job:worker/task:{t}/device:GPU:{g}') \
              for t in worker_indices for g in xrange(self._num_gpus)]

  def param(self, key, default=None, env=None, parser=None):
    r'''Get param for key with default value.
    '''
    if env is None:
      return self._params.get(key, default)
    if parser is None:
      parser = lambda s: s
    try:
      default_value = parser(os.getenv(env, default))
    except:  # pylint: disable=bare-except
      default_value = default
    return self._params.get(key, default_value)

  def update_params(self, **kwargs):
    r'''Update parameters.
    '''
    for k, v in kwargs.items():
      self._params[k] = v

  def has_param(self, key):
    r'''Has specific param in the conetxt.
    '''
    return key in self._params

  @property
  def training_hooks(self):
    r'''Get all training hooks.
    '''
    return self._training_hooks

  @property
  def training_chief_hooks(self):
    r'''Get all training chief hooks.
    '''
    return self._training_chief_hooks

  @property
  def evaluation_hooks(self):
    r'''Get all evaluation hooks.
    '''
    return self._evaluation_hooks

  @property
  def prediction_hooks(self):
    r'''Get all prediction hooks.
    '''
    return self._prediction_hooks

  def add_training_hook(self, hook):
    r'''Add training hook.
    '''
    self._training_hooks.append(hook)

  def add_training_chief_hook(self, hook):
    r'''Add training chief hook.
    '''
    self._training_chief_hooks.append(hook)

  def add_evaluation_hook(self, hook):
    r'''Add evaluation hook.
    '''
    self._evaluation_hooks.append(hook)

  def add_prediction_hook(self, hook):
    r'''Add prediction hook.
    '''
    self._prediction_hooks.append(hook)

context = Context.get()

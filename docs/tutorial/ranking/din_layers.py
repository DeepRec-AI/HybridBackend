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

r'''Keras Layers for DIN model.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend as K

try:
  Unicode = unicode
except NameError:
  Unicode = str


class Concat(tf.keras.layers.Layer):
  r'''Concatenate embeddings from different features.
  '''
  def __init__(self, axis, supports_masking=True, **kwargs):
    super().__init__(**kwargs)
    self.axis = axis
    self.supports_masking = supports_masking

  def call(self, inputs):
    return tf.concat(inputs, axis=self.axis)

  def compute_mask(self, inputs, mask=None):
    r'''Compute mask for concatenation.
    '''
    if not self.supports_masking:
      return None
    if mask is None:
      # pylint: disable=protected-access
      mask = [inputs_i._keras_mask if hasattr(inputs_i, '_keras_mask')
              else None for inputs_i in inputs]
    if mask is None:
      return None
    if not isinstance(mask, list):
      raise ValueError('`mask` should be a list.')
    if not isinstance(inputs, list):
      raise ValueError('`inputs` should be a list.')
    if len(mask) != len(inputs):
      raise ValueError('The lists `inputs` and `mask` '
                       'should have the same length.')
    if all((m is None for m in mask)):
      return None
    # Make a list of masks while making sure
    # the dimensionality of each mask
    # is the same as the corresponding input.
    masks = []
    for input_i, mask_i in zip(inputs, mask):
      if mask_i is None:
        # Input is unmasked. Append all 1s to masks,
        masks.append(tf.ones_like(input_i, dtype='bool'))
      elif K.ndim(mask_i) < K.ndim(input_i):
        # Mask is smaller than the input, expand it
        masks.append(tf.expand_dims(mask_i, axis=-1))
      else:
        masks.append(mask_i)
    concatenated = K.concatenate(masks, axis=self.axis)
    return K.all(concatenated, axis=-1, keepdims=False)

  def get_config(self, ):
    config = {'axis': self.axis, 'supports_masking': self.supports_masking}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class NoMask(tf.keras.layers.Layer):
  r'''A layer having no mask on inputs.
  '''
  def call(self, x, mask=None, **kwargs):  # pylint: disable=unused-argument
    return x

  def compute_mask(self, inputs, mask):  # pylint: disable=unused-argument
    return None


def concat_func(inputs, axis=-1, mask=False):
  if len(inputs) == 1:
    input_first = inputs[0]
    if not mask:
      input_first = NoMask()(input_first)
    return input_first
  return Concat(axis, supports_masking=mask)(inputs)


class Dice(tf.keras.layers.Layer):
  r'''The Data Adaptive Activation Function in DIN,
  which can be viewed as a generalization of PReLu and can adaptively
  adjust the rectified point according to distribution of input data.

  Input shape
    - Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape
    - Same shape as the input.

  Arguments
    - **axis** : Integer, the axis that should be used to
      compute data distribution (typically the features axis).
    - **epsilon** : Small float added to variance to avoid dividing by zero.
  '''

  def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
    self.axis = axis
    self.epsilon = epsilon
    super().__init__(**kwargs)

  def build(self, input_shape):
    self.bn = tf.keras.layers.BatchNormalization(
      axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
    self.alphas = self.add_weight(
      shape=(input_shape[-1],),
      initializer=tf.keras.initializers.Zeros(),
      dtype=tf.float32, name='dice_alpha')  # name='alpha_'+self.name
    super().build(input_shape)  # Be sure to call this somewhere!
    self.uses_learning_phase = True

  def call(self, inputs, training=None, **kwargs):  # pylint: disable=unused-argument
    inputs_normed = self.bn(inputs, training=training)
    x_p = tf.sigmoid(inputs_normed)
    return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self, ):
    config = {'axis': self.axis, 'epsilon': self.epsilon}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def activation_layer(activation):
  r'''Build the activation layer.
  '''
  if activation in ('dice', 'Dice'):
    act_layer = Dice()
  elif isinstance(activation, (str, Unicode)):
    act_layer = tf.keras.layers.Activation(activation)
  elif issubclass(activation, tf.keras.laeyrs.Layer):
    act_layer = activation()
  else:
    raise ValueError(
      f'Invalid activation,found {activation}.'
      'You should use a str or a Activation Layer Class.')
  return act_layer


class DNN(tf.keras.layers.Layer):
  r'''The Multi Layer Percetron.

  Input shape
    - nD tensor with shape: ``(batch_size, ..., input_dim)``.
      The most common situation would be a 2D
      input with shape ``(batch_size, input_dim)``.

  Output shape
    - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
      For instance, for a 2D input with shape ``(batch_size, input_dim)``,
      the output would have shape ``(batch_size, hidden_size[-1])``.

  Arguments
    - **hidden_units**:list of positive integer,
      the layer number and units in each layer.
    - **activation**: Activation function to use.
    - **l2_reg**: float between 0 and 1. L2 regularizer strength applied
      to the kernel weights matrix.
    - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
    - **use_bn**: bool. Whether use BatchNormalization before activation or not.
    - **output_activation**: Activation function to use in the last layer.
      If ``None``,it will be same as ``activation``.
    - **seed**: A Python integer to use as random seed.
  '''

  def __init__(
    self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0,
      use_bn=False, output_activation=None, seed=1024, **kwargs):
    self.hidden_units = hidden_units
    self.activation = activation
    self.l2_reg = l2_reg
    self.dropout_rate = dropout_rate
    self.use_bn = use_bn
    self.output_activation = output_activation
    self.seed = seed

    super().__init__(**kwargs)

  def build(self, input_shape):
    r'''Build the dnn layer.
    '''
    input_size = input_shape[-1]
    hidden_units = [int(input_size)] + list(self.hidden_units)
    self.kernels = [
      self.add_weight(
        name='kernel' + str(i),
        shape=(hidden_units[i], hidden_units[i + 1]),
        initializer=tf.keras.initializers.glorot_normal(
          seed=self.seed),
        regularizer=tf.keras.regularizers.l2(self.l2_reg),
        trainable=True) for i in range(len(self.hidden_units))]
    self.bias = [
      self.add_weight(
        name='bias' + str(i),
        shape=(self.hidden_units[i],),
        initializer=tf.keras.initializers.Zeros(),
        trainable=True) for i in range(len(self.hidden_units))]
    if self.use_bn:
      self.bn_layers = [
        tf.keras.layers.BatchNormalization() for _ in range(
          len(self.hidden_units))]

    self.dropout_layers = [
      tf.keras.layers.Dropout(
        self.dropout_rate, seed=self.seed + i) for i in range(
          len(self.hidden_units))]

    self.activation_layers = [
      activation_layer(self.activation) for _ in range(len(self.hidden_units))]

    if self.output_activation:
      self.activation_layers[-1] = activation_layer(self.output_activation)

    super().build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs, training=None, **kwargs):  # pylint: disable=unused-argument
    r'''Invoke the dnn layer.
    '''
    deep_input = inputs

    for i in range(len(self.hidden_units)):
      fc = tf.nn.bias_add(tf.tensordot(
        deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

      if self.use_bn:
        fc = self.bn_layers[i](fc, training=training)
      try:
        fc = self.activation_layers[i](fc, training=training)
      except TypeError as e:
        print(
          f'make sure the activation function use training flag properly {e}')
        fc = self.activation_layers[i](fc)

      fc = self.dropout_layers[i](fc, training=training)
      deep_input = fc

    return deep_input

  def compute_output_shape(self, input_shape):
    if len(self.hidden_units) > 0:
      shape = input_shape[:-1] + (self.hidden_units[-1],)
    else:
      shape = input_shape
    return tuple(shape)

  def get_config(self, ):
    config = {'activation': self.activation, 'hidden_units': self.hidden_units,
              'l2_reg': self.l2_reg, 'use_bn': self.use_bn,
              'dropout_rate': self.dropout_rate,
              'output_activation': self.output_activation, 'seed': self.seed}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class LocalActivationUnit(tf.keras.layers.Layer):
  r'''The LocalActivationUnit used in DIN with which the representation of
  user interests varies adaptively given different candidate items.

  Input shape
    - A list of two 3D tensor with shape:
    ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

  Output shape
    - 3D tensor with shape: ``(batch_size, T, 1)``.

  Arguments
    - **hidden_units**:list of positive integer,
      the attention net layer number and units in each layer.
    - **activation**: Activation function to use in attention net.
    - **l2_reg**: float between 0 and 1. L2 regularizer strength applied
      to the kernel weights matrix of attention net.
    - **dropout_rate**: float in [0,1). Fraction of
      the units to dropout in attention net.
    - **use_bn**: bool. Whether use BatchNormalization before activation
      or not in attention net.
    - **seed**: A Python integer to use as random seed.
  '''

  def __init__(
    self, hidden_units=(64, 32), activation='sigmoid', l2_reg=0,
      dropout_rate=0, use_bn=False, seed=1024, **kwargs):
    self.hidden_units = hidden_units
    self.activation = activation
    self.l2_reg = l2_reg
    self.dropout_rate = dropout_rate
    self.use_bn = use_bn
    self.seed = seed
    super().__init__(**kwargs)
    self.supports_masking = True

  def build(self, input_shape):
    r'''Build local activation unit.
    '''
    if not isinstance(input_shape, list) or len(input_shape) != 2:
      raise ValueError(
        'A `LocalActivationUnit` layer should be called '
        'on a list of 2 inputs')
    if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
      raise ValueError(
        f'Unexpected inputs dimensions {len(input_shape[0])} '
        f'and {len(input_shape[1])}, expect to be 3 dimensions')

    if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
      raise ValueError(
        'A `LocalActivationUnit` layer requires '
        'inputs of a two inputs with shape (None,1,embedding_size) and '
        '(None,T,embedding_size) Got different shapes: '
        f'{input_shape[0]},{input_shape[1]}')

    size = 4 * int(input_shape[0][-1]) if len(self.hidden_units) == 0\
      else self.hidden_units[-1]
    self.kernel = self.add_weight(
      shape=(size, 1),
      initializer=tf.keras.initializers.glorot_normal(seed=self.seed),
      name='kernel')
    self.bias = self.add_weight(
      shape=(1,), initializer=tf.keras.initializers.Zeros(), name='bias')
    self.dnn = DNN(
      self.hidden_units, self.activation, self.l2_reg,
      self.dropout_rate, self.use_bn, seed=self.seed)

    super().build(input_shape)

  def call(self, inputs, training=None, **kwargs):  # pylint: disable=unused-argument
    r'''Invoke the LocalActivationUnit.
    '''
    query, keys = inputs

    keys_len = keys.get_shape()[1]
    queries = K.repeat_elements(query, keys_len, 1)

    att_input = tf.concat(
      [queries, keys, queries - keys, queries * keys], axis=-1)

    att_out = self.dnn(att_input, training=training)

    attention_score = tf.nn.bias_add(
      tf.tensordot(att_out, self.kernel, axes=(-1, 0)), self.bias)

    return attention_score

  def compute_output_shape(self, input_shape):
    return input_shape[1][:2] + (1,)

  def compute_mask(self, inputs, mask):  # pylint: disable=unused-argument
    return mask

  def get_config(self, ):
    config = {'activation': self.activation, 'hidden_units': self.hidden_units,
              'l2_reg': self.l2_reg, 'dropout_rate': self.dropout_rate,
              'use_bn': self.use_bn, 'seed': self.seed}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class AttentionSequencePoolingLayer(tf.keras.layers.Layer):
  r'''The Attentional sequence pooling operation used in DIN.

  Input shape
    - A list of three tensor: [query,keys,keys_length]
    - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``
    - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
    - keys_length is a 2D tensor with shape: ``(batch_size, 1)``

  Output shape
    - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

  Arguments
    - **att_hidden_units**: list of positive integer,
      the attention net layer number and units in each layer.
    - **att_activation**: Activation function to use in attention net.
    - **weight_normalization**: bool.Whether normalize the attention
      score of local activation unit.
    - **supports_masking**:If True,the input need to support masking.
  '''

  def __init__(
    self, att_hidden_units=(80, 40),
    att_activation='sigmoid', weight_normalization=False,
    return_score=False,
      supports_masking=False, **kwargs):
    self.att_hidden_units = att_hidden_units
    self.att_activation = att_activation
    self.weight_normalization = weight_normalization
    self.return_score = return_score
    super().__init__(**kwargs)
    self.supports_masking = supports_masking

  def build(self, input_shape):
    r'''Build the attention layer.
    '''
    if not self.supports_masking:
      if not isinstance(input_shape, list) or len(input_shape) != 3:
        raise ValueError(
          'A `AttentionSequencePoolingLayer` layer should be called '
          'on a list of 3 inputs')

      if (len(input_shape[0]) != 3
          or len(input_shape[1]) != 3
          or len(input_shape[2]) != 2):
        raise ValueError(
          'Unexpected inputs dimensions, the 3 tensor dimensions are'
          f'{len(input_shape[0])}, {len(input_shape[1])} and '
          f'{len(input_shape[2])}, expect to be 3,3 and 2')

      if (input_shape[0][-1] != input_shape[1][-1]
          or input_shape[0][1] != 1
          or input_shape[2][1] != 1):
        raise ValueError(
          'A `AttentionSequencePoolingLayer` layer requires '
          'inputs of a 3 tensor with shape (None,1,embedding_size),'
          '(None,T,embedding_size) and (None,1)'
          f'Got different shapes: {input_shape}')
    else:
      pass
    self.local_att = LocalActivationUnit(
      self.att_hidden_units, self.att_activation,
      l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,)
    super().build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs, mask=None, training=None, **kwargs):  # pylint: disable=unused-argument
    r'''Invoke the AttentionSequencePoolingLayer.
    '''
    if self.supports_masking:
      if mask is None:
        raise ValueError(
          'When supports_masking=True,input must support masking')
      queries, keys = inputs
      key_masks = tf.expand_dims(mask[-1], axis=1)
    else:
      queries, keys, keys_length = inputs
      hist_len = keys.get_shape()[1]
      key_masks = tf.sequence_mask(keys_length, hist_len)

    attention_score = self.local_att([queries, keys], training=training)
    outputs = tf.transpose(attention_score, (0, 2, 1))

    if self.weight_normalization:
      paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    else:
      paddings = tf.zeros_like(outputs)
    outputs = tf.where(key_masks, outputs, paddings)

    if self.weight_normalization:
      outputs = tf.nn.softmax(outputs)

    if not self.return_score:
      outputs = tf.matmul(outputs, keys)

    # pylint: disable=protected-access
    if tf.__version__ < '1.13.0':
      outputs._uses_learning_phase = attention_score._uses_learning_phase
    else:
      outputs._uses_learning_phase = training is not None

    return outputs

  def compute_output_shape(self, input_shape):
    if self.return_score:
      return (None, 1, input_shape[1][1])
    return (None, 1, input_shape[0][-1])

  def compute_mask(self, inputs, mask):  # pylint: disable=unused-argument
    return None

  def get_config(self, ):
    config = {'att_hidden_units': self.att_hidden_units,
              'att_activation': self.att_activation,
              'weight_normalization': self.weight_normalization,
              'return_score': self.return_score,
              'supports_masking': self.supports_masking}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def combined_dnn_input(sparse_embedding_list, dense_value_list):
  r'''Combine numeric and categorical embeddings.
  '''
  if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
    sparse_dnn_input = tf.keras.layers.Flatten()(
      concat_func(sparse_embedding_list))
    dense_dnn_input = tf.keras.layers.Flatten()(
      concat_func(dense_value_list))
    return concat_func([sparse_dnn_input, dense_dnn_input])
  if len(sparse_embedding_list) > 0:
    return tf.keras.layers.Flatten()(concat_func(sparse_embedding_list))
  if len(dense_value_list) > 0:
    return tf.keras.layers.Flatten()(concat_func(dense_value_list))
  raise NotImplementedError('dnn_feature_columns can not be empty list')


class PredictionLayer(tf.keras.layers.Layer):
  r'''
  Arguments
   - **task**: str, ``"binary"`` for  binary logloss or
     ``"regression"`` for regression loss
   - **use_bias**: bool.Whether add bias term or not.
  '''

  def __init__(self, task='binary', use_bias=True, **kwargs):
    if task not in ['binary', 'multiclass', 'regression']:
      raise ValueError('task must be binary,multiclass or regression')
    self.task = task
    self.use_bias = use_bias
    super().__init__(**kwargs)

  def build(self, input_shape):
    if self.use_bias:
      self.global_bias = self.add_weight(
        shape=(1,), initializer=tf.keras.initializers.Zeros(),
        name='global_bias')
    super().build(input_shape)

  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    x = inputs
    if self.use_bias:
      x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
    if self.task == 'binary':
      x = tf.sigmoid(x)
    output = tf.reshape(x, (-1, 1))
    return output

  def compute_output_shape(self, input_shape):  # pylint: disable=unused-argument
    return (None, 1)

  def get_config(self, ):
    config = {'task': self.task, 'use_bias': self.use_bias}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

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

r'''View of computed tensors.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.util import compat


class OperationLike:
  r'''Declares an operation-like function.
  '''
  @classmethod
  def build_attr_value(cls, value):
    r'''Build attribute value.
    '''
    if isinstance(value, str):
      return attr_value_pb2.AttrValue(s=compat.as_bytes(value))
    if isinstance(value, int):
      return attr_value_pb2.AttrValue(i=value)
    if isinstance(value, float):
      return attr_value_pb2.AttrValue(f=value)
    if isinstance(value, bool):
      return attr_value_pb2.AttrValue(b=value)
    if isinstance(value, dtypes.DType):
      return attr_value_pb2.AttrValue(type=value.as_datatype_enum)
    if isinstance(value, tensor_shape.TensorShape):
      return attr_value_pb2.AttrValue(shape=value.as_proto())
    if isinstance(value, (list, tuple)):
      if not value:
        raise ValueError(f'Attribute {value} is an empty list')
      value = list(value)
      if isinstance(value[0], str):
        for vi in value[1:]:
          if not isinstance(vi, str):
            raise ValueError(f'List attribute {value} is invalid')
        return attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
            s=[compat.as_bytes(vi) for vi in value]))
      if isinstance(value[0], int):
        for vi in value[1:]:
          if not isinstance(vi, str):
            raise ValueError(f'List attribute {value} is invalid')
        return attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(i=value))
      if isinstance(value[0], float):
        for vi in value[1:]:
          if not isinstance(vi, float):
            raise ValueError(f'List attribute {value} is invalid')
        return attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(f=value))
      if isinstance(value[0], bool):
        for vi in value[1:]:
          if not isinstance(vi, bool):
            raise ValueError(f'List attribute {value} is invalid')
        return attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(b=value))
      if isinstance(value[0], dtypes.DType):
        for vi in value[1:]:
          if not isinstance(vi, dtypes.DType):
            raise ValueError(f'List attribute {value} is invalid')
        return attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
            type=[vi.as_datatype_enum for vi in value]))
      if isinstance(value[0], tensor_shape.TensorShape):
        for vi in value[1:]:
          if not isinstance(vi, tensor_shape.TensorShape):
            raise ValueError(f'List attribute {value} is invalid')
        return attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
            type=[vi.as_proto() for vi in value]))
      raise ValueError(f'List attribute {value} is invalid')
    raise ValueError(f'Attribute {value} is invalid')

  @classmethod
  def build_attr_dict(cls, op_name, default_attr, **kwargs):
    r'''Build attribute key and value.
    '''
    attrs = {}
    if default_attr is not None:
      attrs[f'_HB_{op_name}'] = cls.build_attr_value(default_attr)
    for k, v in kwargs.items():
      attrs[f'_HB_{op_name}_{k}'] = cls.build_attr_value(v)
    return attrs

  def __init__(self, name, graph=None):
    self._name = name
    self._graph = ops.get_default_graph() if graph is None else graph
    self._tensor_specs = []
    self._is_single_tensor = False

  def returns_tensor(self, shape, dtype):
    r'''Set single tensor specification.
    '''
    self._tensor_specs = [tensor_spec.TensorSpec(shape, dtype)]
    self._is_single_tensor = True
    return self

  def returns_tensors(self, *tensor_specs):
    r'''Set tensor specifications.
    '''
    for i, o in enumerate(tensor_specs):
      if not isinstance(o, tensor_spec.TensorSpec):
        raise ValueError(f'Argument {i} is not a tf.TensorSpec')
    self._tensor_specs = tensor_specs
    self._is_single_tensor = False
    return self

  def finalize(self, *args, **kwargs):
    r'''Finalizes this operation-like in graph.
    '''
    name = kwargs.pop('name', None)
    if name is None:
      name = self._graph.unique_name(self._name)
    attrs = OperationLike.build_attr_dict(self._name, True, **kwargs)

    if not args:
      if not self._tensor_specs:
        with self._graph._attr_scope(attrs):  # pylint: disable=protected-access
          return control_flow_ops.no_op(name=name)

      notinputs = [
        gen_functional_ops.fake_param(
          dtype=o.dtype,
          shape=o.shape,
          name=f'{name}_notinput{i}')
        for i, o in enumerate(self._tensor_specs)]
      with self._graph._attr_scope(attrs):  # pylint: disable=protected-access
        outputs = array_ops.identity_n(notinputs, name=name)
        return outputs[0] if self._is_single_tensor else outputs

    proxied_inputs = []
    for idx, arg in enumerate(args):
      with self._graph._attr_scope(  # pylint: disable=protected-access
          OperationLike.build_attr_dict(self._name, None, input_proxy=idx)):
        if isinstance(arg, (list, tuple)):
          proxied_inputs.extend(
            array_ops.identity_n(arg, name=f'{name}_input{idx}'))
        else:
          proxied_inputs.append(
            array_ops.identity(arg, name=f'{name}_input{idx}'))

    if not self._tensor_specs:
      with ops.control_dependencies(proxied_inputs):
        with self._graph._attr_scope(attrs):  # pylint: disable=protected-access
          return control_flow_ops.no_op(name=name)

    notinputs = [
      gen_functional_ops.fake_param(
        dtype=o.dtype,
        shape=o.shape,
        name=f'{name}_notinput{i}')
      for i, o in enumerate(self._tensor_specs)]
    with ops.control_dependencies(proxied_inputs):
      with self._graph._attr_scope(attrs):  # pylint: disable=protected-access
        outputs = array_ops.identity_n(notinputs, name=name)
        return outputs[0] if self._is_single_tensor else outputs

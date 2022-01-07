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

r'''SavedModel builder implementation.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.builder_impl import SavedModelBuilder \
    as _SavedModelBuilder
from tensorflow.python.ops import variables

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.training.saver import HybridBackendSaverBase
from hybridbackend.tensorflow.training.saver import Saver


class SavedModelBuilder(_SavedModelBuilder):
  r'''Builds the `SavedModel` protocol buffer and saves variables and assets.
  '''
  def __init__(self, export_dir):
    super().__init__(export_dir=export_dir)
    self._rank =  Context.get().rank

  def _save_and_write_assets(self, assets_collection_to_add=None):
    r'''Saves asset to the meta graph and writes asset files to disk.
    '''
    if self._rank == 0:
      super()._save_and_write_assets(
          assets_collection_to_add=assets_collection_to_add)

  def _maybe_create_saver(self, saver=None):
    r'''Creates a sharded saver if one does not already exist.
    '''
    if not saver:
      saver = Saver(
          variables._all_saveable_objects(),  # pylint: disable=protected-access
          sharded=True,
          write_version=saver_pb2.SaverDef.V2,
          allow_empty=True)
    if not isinstance(saver, HybridBackendSaverBase):
      raise ValueError('saver must be hb.train.Saver')
    return saver

  def _tag_and_add_meta_graph(self, meta_graph_def, tags, signature_def_map):
    r'''Tags the meta graph def and adds it to the SavedModel.
    '''
    if self._rank == 0:
      super()._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

  def save(self, as_text=False):
    r'''Writes a `SavedModel` protocol buffer to disk.
    '''
    if self._rank == 0:
      super().save(as_text=as_text)

  def add_meta_graph_and_variables(
      self, sess, tags,
      signature_def_map=None,
      assets_collection=None,
      legacy_init_op=None,
      clear_devices=False,
      main_op=None,
      strip_default_attrs=False,
      saver=None):
    r'''Add meta graph and variables.
    '''
    if self._has_saved_variables:  # pylint: disable=access-member-before-definition
      raise AssertionError("Graph state including variables and assets has "
                           "already been saved. Please invoke "
                           "`add_meta_graph()` instead.")

    signature_def_map = signature_def_map or {}
    self._validate_signature_def_map(signature_def_map)
    main_op = main_op or legacy_init_op
    self._add_collections(assets_collection, main_op, None)

    saved_model_utils.get_or_create_variables_dir(self._export_dir)
    variables_path = saved_model_utils.get_variables_path(self._export_dir)

    saver = self._maybe_create_saver(saver)
    saver.save(sess, variables_path, write_meta_graph=False, write_state=False)
    if self._rank == 0:
      meta_graph_def = saver.export_meta_graph(
          clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)
      self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)
    self._has_saved_variables = True

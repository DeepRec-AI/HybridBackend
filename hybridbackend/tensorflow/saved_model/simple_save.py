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

r'''SavedModel export functionality.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.training import monitored_session
from tensorflow.python.util import compat

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.device import device_function
from hybridbackend.tensorflow.framework.scope import scope
from hybridbackend.tensorflow.saved_model.builder_impl import SavedModelBuilder
from hybridbackend.tensorflow.training.saver import Saver
from hybridbackend.tensorflow.training.server_lib import build_session_config


def export(
    export_dir_base, checkpoint_path, fn,
    assets_extra=None,
    as_text=False,
    clear_devices=True,
    strip_default_attrs=True):
  r'''Build a SavedModel from variables in checkpoint.

  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
    checkpoint_path: A path to a checkpoint.
    fn: Function constructs graph to save and returns a signature_def.
    assets_extra: A dict specifying how to populate the assets.extra directory
        within the exported SavedModel.  Each key should give the destination
        path (including the filename) relative to the assets.extra directory.
        The corresponding value gives the full path of the source file to be
        copied.  For example, the simple case of copying a single file without
        renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
    as_text: Whether or not to write the SavedModel proto in text format.
    clear_devices: Whether or not to clear the device field.
    strip_default_attrs: Whether or not to remove default-valued attributes
        from the NodeDefs.
  '''
  export_dir = export_utils.get_timestamped_export_dir(export_dir_base)
  export_basename = os.path.basename(compat.as_bytes(export_dir))
  temp_export_dir = os.path.join(
      compat.as_bytes(export_dir_base),
      compat.as_bytes(
          f'temp-{Context.get().rank}-{compat.as_text(export_basename)}'))

  with ops.Graph().as_default(), \
      ops.device(device_function), \
      scope(comm_pool_name=export_basename):
    signature_def_map = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: fn()}

    with tf_session.Session(
        Context.get().target, config=build_session_config(None)) as sess:
      local_init_op = monitored_session.Scaffold.default_local_init_op()
      saver = Saver()
      try:
        saver.restore(sess, checkpoint_path)
      except errors.NotFoundError as e:
        msg = ('Could not load all requested variables from checkpoint.'
               ' Please make sure your fn does not expect '
               'variables that were not saved in the checkpoint.\n\n'
               'Encountered error while restoring checkpoint from: '
               f'`{checkpoint_path}`. Full Traceback:\n\n{e}')
        raise ValueError(msg) from e

      gfile.MakeDirs(temp_export_dir)
      b = SavedModelBuilder(temp_export_dir)
      b.add_meta_graph_and_variables(
          sess,
          tags=[tag_constants.SERVING],
          signature_def_map=signature_def_map,
          assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
          main_op=local_init_op,
          saver=saver,
          clear_devices=clear_devices,
          strip_default_attrs=strip_default_attrs)
      b.save(as_text)

      if Context.get().rank == 0:
        if assets_extra:
          assets_extra_path = os.path.join(compat.as_bytes(temp_export_dir),
                                           compat.as_bytes('assets.extra'))
          for dest_relative, source in assets_extra.items():
            dest_absolute = os.path.join(compat.as_bytes(assets_extra_path),
                                         compat.as_bytes(dest_relative))
            dest_path = os.path.dirname(dest_absolute)
            gfile.MakeDirs(dest_path)
            gfile.Copy(source, dest_absolute)

        gfile.Rename(temp_export_dir, export_dir)
      else:
        variables_path = os.path.join(
            compat.as_bytes(temp_export_dir), b'variables')
        if not gfile.ListDirectory(variables_path):
          gfile.DeleteRecursively(variables_path)
        if not gfile.ListDirectory(temp_export_dir):
          gfile.DeleteRecursively(temp_export_dir)

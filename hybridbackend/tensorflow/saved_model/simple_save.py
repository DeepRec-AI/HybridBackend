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
import re

from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.mode_keys import \
  KerasModeKeys as ModeKeys
from tensorflow.python.training import monitored_session
from tensorflow.python.util import compat

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.scope import scope


def export_all(
    export_dir_base,
    checkpoint_path,
    signature_defs_and_main_op_fn,
    assets_extra=None,
    as_text=False,
    clear_devices=True,
    strip_default_attrs=True,
    modes=None):
  r'''Build a SavedModel from variables in checkpoint.

  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
    checkpoint_path: A path to a checkpoint.
    signature_defs_and_main_op_fn: Function returns signature defs and main_op.
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
    modes: List contains PREDICT, TRAIN or TEST.

  Returns:
    Export directory if it's chief.
  '''
  if Context.get().rank != 0:
    return None

  export_dir = export_utils.get_timestamped_export_dir(export_dir_base)
  with ops.Graph().as_default(), scope(enable_sharding=False):
    # Build graph.
    signature_def_map = signature_defs_and_main_op_fn()
    main_op = None
    if isinstance(signature_def_map, (tuple, list)):
      if len(signature_def_map) > 1:
        main_op = signature_def_map[1]
      signature_def_map = signature_def_map[0]
    if not main_op:
      main_op = monitored_session.Scaffold.default_local_init_op()
    if modes is None:
      modes = [ModeKeys.PREDICT, ModeKeys.TRAIN, ModeKeys.TEST]
    modes = [
      m for m in modes
      if export_utils.SIGNATURE_KEY_MAP[m] in signature_def_map]
    signature_def_map = {
      k: signature_def_map[k] for k in signature_def_map
      if k in [export_utils.SIGNATURE_KEY_MAP[m] for m in modes]}
    signature_tags = [export_utils.EXPORT_TAG_MAP[m][0] for m in modes]

    b = builder.SavedModelBuilder(export_dir)
    b._has_saved_variables = True  # pylint: disable=protected-access

    # Copy variables.
    saved_model_utils.get_or_create_variables_dir(export_dir)
    export_checkpoint_path = saved_model_utils.get_variables_path(export_dir)
    checkpoint_files = [
      *gfile.Glob(f'{checkpoint_path}.index'),
      *gfile.Glob(f'{checkpoint_path}.data-?????-of-?????')]
    for f in checkpoint_files:
      export_ckpt = re.sub(
        compat.as_text(checkpoint_path),
        compat.as_text(export_checkpoint_path),
        f)
      gfile.Copy(f, export_ckpt)

    # Add MetaGraph.
    b.add_meta_graph(
      tags=signature_tags,
      signature_def_map=signature_def_map,
      assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
      clear_devices=clear_devices,
      main_op=main_op,
      strip_default_attrs=strip_default_attrs)

    # Save model.
    b.save(as_text=as_text)

    # Save extras.
    if assets_extra:
      assets_extra_path = os.path.join(
        export_dir, constants.EXTRA_ASSETS_DIRECTORY)
      for dst, src in assets_extra.items():
        target = os.path.join(assets_extra_path, compat.as_bytes(dst))
        gfile.MakeDirs(os.path.dirname(target))
        gfile.Copy(src, target)

  return export_dir


def export(
    export_dir_base,
    checkpoint_path,
    signature_def_fn,
    assets_extra=None,
    as_text=False,
    clear_devices=True,
    strip_default_attrs=True,
    mode=ModeKeys.PREDICT):
  r'''Build a SavedModel from variables in checkpoint.

  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
    checkpoint_path: A path to a checkpoint.
    signature_def_fn: Function returns a signature_def.
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
    mode: PREDICT, TRAIN ot TEST

  Returns:
    Export directory if it's chief.
  '''
  return export_all(
    export_dir_base,
    checkpoint_path,
    lambda: {export_utils.SIGNATURE_KEY_MAP[mode]: signature_def_fn()},
    assets_extra=assets_extra,
    as_text=as_text,
    clear_devices=clear_devices,
    strip_default_attrs=strip_default_attrs,
    modes=[mode])

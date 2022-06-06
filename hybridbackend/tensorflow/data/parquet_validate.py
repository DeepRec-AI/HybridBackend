#!/usr/bin/env python

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

r'''Validates Parquet files.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hybridbackend.tensorflow as hb
import os
import sys
import tensorflow as tf


def _main(args):
  r'''Entry function.
  '''
  if not args.input_files:
    return
  num_files = len(args.input_files)
  sys.stderr.write(f'[Parquet] Validating schema of {num_files} files...\n')
  fields = None
  if args.fields:
    fields = args.fields.split(',')
  prev_fields = None
  for f in args.input_files:
    actual_fields = hb.data.ParquetDataset.read_schema(
      f, fields=fields, lower=args.lower)
    if prev_fields:
      if len(actual_fields) != len(prev_fields):
        raise ValueError(
          f'{f} has {len(actual_fields)} fields while '
          f'{args.input_files[0]} has {len(prev_fields)} fields')
      prev_f = args.input_files[0]
      for idx, field in enumerate(actual_fields):
        prev_field = prev_fields[idx]
        if field.name != prev_field.name:
          raise ValueError(
            f'name of field {idx} of {f} and {prev_f} are different: '
            f'{field.name} vs {prev_field.name}')
        if field.dtype != prev_field.dtype:
          raise ValueError(
            f'dtype of field {idx} of {f} and {prev_f} are different: '
            f'{field.dtype} vs {prev_field.dtype}')
        if field.ragged_rank != prev_field.ragged_rank:
          raise ValueError(
            f'ragged_rank of field {idx} of {f} and {prev_f} are different: '
            f'{field.ragged_rank} vs {prev_field.ragged_rank}')
        if field.shape != prev_field.shape:
          raise ValueError(
            f'shape of field {idx} of {f} and {prev_f} are different: '
            f'{field.shape} vs {prev_field.shape}')
    else:
      prev_fields = actual_fields
  sys.stderr.write('[Parquet] Schema validated.\n')

  with tf.Graph().as_default() as graph:
    peek_ops = []
    for f in args.input_files:
      ds = hb.data.ParquetDataset(f, batch_size=args.peek, fields=prev_fields)
      batch = hb.data.make_one_shot_iterator(ds).get_next()
      peek_ops.append(batch)

    with tf.Session(graph=graph) as sess:
      for fidx, f in enumerate(args.input_files):
        sys.stderr.write(
          f'[Parquet] Validating {f} ({fidx + 1} of {num_files})...\n')
        sess.run(peek_ops[fidx])
        sys.stderr.write(f'[Parquet] {f} validated.\n')


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  parser = argparse.ArgumentParser()
  parser.add_argument('input_files', nargs='+')
  parser.add_argument('--fields', type=str, default='')
  parser.add_argument('--lower', default=False, action='store_true')
  parser.add_argument('--peek', type=int, default=32)
  _main(parser.parse_args())

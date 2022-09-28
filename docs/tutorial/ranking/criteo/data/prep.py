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

r'''Prepare Criteo 1TB Click Logs Dataset.

See https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/ for more
information.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm


def main(args):
  label_names = [args.label_prefix]
  if_names = [f'{args.integer_features_prefix}{i}' for i in range(13)]
  cf_names = [f'{args.categorical_features_prefix}{i}' for i in range(26)]

  pa_schema = pa.schema(
    [(n, pa.int32()) for n in label_names]
    + [(n, pa.int32()) for n in if_names]
    + [(n, pa.int64()) for n in cf_names])
  pd_schema = dict(
    [(n, np.int32) for n in label_names]
    + [(n, np.int32) for n in if_names]
    + [(n, np.int64) for n in cf_names]
  )

  converters = dict(
    [(n, np.int32) for n in label_names]
    + [(n, lambda i: int(i) if i else args.null_value) for n in if_names]
    + [(n, lambda i: int(i, 16) if i else args.null_value) for n in cf_names]
  )

  parquet_fname = f'{os.path.splitext(args.fname)[0]}.parquet'
  try:
    with pq.ParquetWriter(
        parquet_fname, pa_schema,
        use_dictionary=not args.no_use_dictionary,
        compression=args.compression,
        flavor=args.flavor) as writer:
      for dfc in tqdm.tqdm(
          pd.read_csv(
            args.fname,
            sep='\t',
            names=label_names + if_names + cf_names,
            converters=converters,
            chunksize=args.row_group_size),
          desc=f'Prepare dataset from {args.fname}',
          unit='blocks'):
        pt = pa.Table.from_pandas(dfc.astype(pd_schema), preserve_index=False)
        writer.write_table(pt)
        del pt
  except Exception:
    warnings.warn(
      f'Failed to prepare dataset from {args.fname}',
      RuntimeWarning)
    raise


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--label-prefix', default='label')
  parser.add_argument('--integer-features-prefix', default='if')
  parser.add_argument('--categorical-features-prefix', default='cf')
  parser.add_argument('--compression', default='zstd')
  parser.add_argument('--flavor', default='spark')
  parser.add_argument('--no-use-dictionary', default=False, action='store_true')
  parser.add_argument('--row-group-size', type=int, default=1000000)
  parser.add_argument('--null-value', type=int, default=-1 << 16)
  parser.add_argument('fname')
  main(parser.parse_args())

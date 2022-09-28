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

r'''Prepare Taobao Click Logs Dataset.

Step 2: Build user behavior logs.

See https://tianchi.aliyun.com/dataset/dataDetail?dataId=56 for more
information.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _aggregate(
    hist, name, output_fname, no_use_dictionary, compression, flavor):
  r'''Aggregate user behavior into lists
  '''
  logging.info('Grouping %s logs...', name)
  del hist['btag']
  hist = hist.rename(
    columns={
      'time_stamp': f'user_{name}_ts_list',
      'cate': f'user_{name}_category_list',
      'brand': f'user_{name}_brand_list'})
  hist = hist.sort_values(by=[f'user_{name}_ts_list'])
  hist = hist.groupby(
    ['user'], as_index=False, sort=False).aggregate(list)
  logging.info('Writing parquet file %s...', output_fname)
  hist_table = pa.Table.from_pandas(hist, preserve_index=False)
  pq.write_table(
    hist_table, output_fname,
    use_dictionary=not no_use_dictionary,
    compression=compression,
    flavor=flavor)
  del hist
  del hist_table


def main(args):
  logging.info('Reading behavior logs from %s...', args.fname)
  behavior_log_dtypes = {
    'time_stamp': np.int64,
    'user': np.int32,
    'btag': str,
    'cate': np.int32,
    'brand': np.int32,
  }
  behavior_log = pd.read_csv(
    args.fname,
    sep=',',
    dtype=behavior_log_dtypes)
  pv_log = behavior_log[behavior_log.btag == 'pv'].copy()
  cart_log = behavior_log[behavior_log.btag == 'cart'].copy()
  fav_log = behavior_log[behavior_log.btag == 'fav'].copy()
  buy_log = behavior_log[behavior_log.btag == 'buy'].copy()
  del behavior_log

  _aggregate(
    pv_log, 'pv', args.pv_output_fname,
    args.no_use_dictionary, args.compression, args.flavor)
  _aggregate(
    cart_log, 'cart', args.cart_output_fname,
    args.no_use_dictionary, args.compression, args.flavor)
  _aggregate(
    fav_log, 'fav', args.fav_output_fname,
    args.no_use_dictionary, args.compression, args.flavor)
  _aggregate(
    buy_log, 'buy', args.buy_output_fname,
    args.no_use_dictionary, args.compression, args.flavor)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--compression', default='zstd')
  parser.add_argument('--flavor', default='spark')
  parser.add_argument('--no-use-dictionary', default=False, action='store_true')
  parser.add_argument('--fname', default='./behavior_log.csv')
  parser.add_argument('--pv-output-fname', default='./pv_log.parquet')
  parser.add_argument('--cart-output-fname', default='./cart_log.parquet')
  parser.add_argument('--fav-output-fname', default='./fav_log.parquet')
  parser.add_argument('--buy-output-fname', default='./buy_log.parquet')
  main(parser.parse_args())

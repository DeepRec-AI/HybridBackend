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

Step 3: Merge user bahvaior logs into backbone click logs.

See https://tianchi.aliyun.com/dataset/dataDetail?dataId=56 for more
information.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

tqdm.tqdm.pandas()


def _merge_to_clicks(click_log, hist, hist_cols):
  r'''Merge history clicks into click logs.
  '''
  merged = click_log.merge(hist, on='user', how='left')
  for col in hist_cols:
    mask = merged[col].isna()
    merged.loc[mask, col] = merged.loc[mask, col].fillna('[]').apply(eval)
  return merged


def _clip_clicks(row, behavior_log_cols, duration):
  r'''Clip history clicks inside click logs.
  '''
  max_ts = row.ts
  min_ts = max_ts - duration
  for ts_col, cols in behavior_log_cols.items():
    ts_list = row[ts_col]
    begin_idx = 0
    end_idx = len(ts_list)
    for ts in ts_list:
      if ts < min_ts:
        begin_idx += 1
      elif ts >= max_ts:
        end_idx -= 1
    for col in cols:
      row[col] = row[col][begin_idx: end_idx]
  return row


def main(args):
  logging.info('Reading backbone click logs from %s...', args.fname)
  click_log = pd.read_parquet(args.fname)

  behavior_logs = {
    'pv': args.pv_log_fname,
    'cart': args.cart_log_fname,
    'fav': args.fav_log_fname,
    'buy': args.buy_log_fname}

  behavior_log_cols = {}
  for btag, fname in behavior_logs.items():
    logging.info('Reading %s logs from %s...', btag, fname)
    hist = pd.read_parquet(fname)
    hist_cols = hist.columns.to_list()
    hist_cols.remove('user')
    logging.info('Merging %s logs into click logs...', btag)
    click_log = _merge_to_clicks(click_log, hist, hist_cols)
    hist_cols.remove(f'user_{btag}_ts_list')
    behavior_log_cols[f'user_{btag}_ts_list'] = hist_cols
    del hist

  logging.info('Clipping click logs...')
  click_log = click_log.progress_apply(
    lambda row: _clip_clicks(row, behavior_log_cols, args.clip_secs),
    axis=1)
  for tscol in behavior_log_cols:
    del click_log[tscol]

  logging.info('Writing parquet file %s...', args.output_fname)
  click_log_table = pa.Table.from_pandas(click_log, preserve_index=False)
  pq.write_table(
    click_log_table, args.output_fname,
    use_dictionary=not args.no_use_dictionary,
    compression=args.compression,
    flavor=args.flavor)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--compression', default='zstd')
  parser.add_argument('--flavor', default='spark')
  parser.add_argument('--no-use-dictionary', default=False, action='store_true')
  parser.add_argument('--clip-secs', type=int, default=86400 * 3)  # 3 days
  parser.add_argument('--pv-log-fname', default='./pv_log.parquet')
  parser.add_argument('--cart-log-fname', default='./cart_log.parquet')
  parser.add_argument('--fav-log-fname', default='./fav_log.parquet')
  parser.add_argument('--buy-log-fname', default='./buy_log.parquet')
  parser.add_argument('fname')
  parser.add_argument('output_fname')
  main(parser.parse_args())

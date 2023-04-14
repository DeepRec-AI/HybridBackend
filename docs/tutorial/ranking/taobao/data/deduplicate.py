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

r'''Deduplicate Taobao Click Logs Dataset.
Re-organize the data of all columns into n rows, where each row is a
list of deduplicated values from `deduplicated_block_size` of original rows.
In addition, columns sharing the same belongs insert a `data_index` column into
the table, where each row of `data_index` has a
`deduplicated_block_size`-length list of indices for a later restoring action.
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
import tqdm

tqdm.tqdm.pandas()


def _gather(
    input_data, input_indices, start):
  output_val = []
  for idx in input_indices:
    output_val.append(input_data[start + idx])
  return output_val


def main(args):
  logging.info('Reading click logs from %s...', args.fname)
  click_log = pd.read_parquet(args.fname)
  users = click_log[args.user_cols[0]].tolist()
  start = 0
  end = args.deduplicated_block_size
  unique_users = []
  unique_users_index = []
  unique_users_inverse = []
  non_users_inverse = []
  while end < len(users):
    val, index, inverse = np.unique(
      users[start:end], return_index=True, return_inverse=True)
    unique_users.append(val)
    unique_users_index.append(index)
    unique_users_inverse.append(inverse)
    non_users_inverse.append(np.arange(end - start))
    start = end
    end += args.deduplicated_block_size

  if start != len(users):
    val, index, inverse = np.unique(
      users[start:len(users)], return_index=True, return_inverse=True)
    unique_users.append(val)
    unique_users_index.append(index)
    unique_users_inverse.append(inverse)
    non_users_inverse.append(np.arange(len(users) - start))

  total_cols = [unique_users_inverse, non_users_inverse, unique_users]
  total_field_names = [
    'users_inverse', 'non_users_inverse'] + args.user_cols + args.non_user_cols
  for col_name in args.user_cols[1:]:
    col_val = click_log[col_name].tolist()
    col_unique_val = []
    for i, user_idx in enumerate(unique_users_index):
      col_unique_val.append(
        _gather(col_val, user_idx, i * args.deduplicated_block_size))
    total_cols.append(col_unique_val)

  for col_name in args.non_user_cols:
    col_val = click_log[col_name].tolist()
    col_unique_val = []
    for i, inverse in enumerate(non_users_inverse):
      start_pos = i * args.deduplicated_block_size
      end_pos = start_pos + len(inverse)
      col_unique_val.append(col_val[start_pos:end_pos])
    total_cols.append(col_unique_val)

  deduplicated_table = pa.Table.from_arrays(
    total_cols, names=total_field_names)
  logging.info('Writing parquet file %s...', args.output_fname)
  pq.write_table(
    deduplicated_table, args.output_fname,
    use_dictionary=not args.no_use_dictionary,
    compression=args.compression,
    flavor=args.flavor)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--compression', default='zstd')
  parser.add_argument('--flavor', default='spark')
  parser.add_argument('--no-use-dictionary', default=False, action='store_true')
  parser.add_argument('--deduplicated-block-size', type=int, default=1000)
  parser.add_argument(
    '--user-cols', nargs='+', default=[
      'user',
      'user_cms_seg',
      'user_cms_group',
      'user_gender',
      'user_age',
      'user_pvalue',
      'user_shopping',
      'user_occupation',
      'user_city',
      'user_pv_category_list',
      'user_cart_category_list',
      'user_fav_category_list',
      'user_buy_category_list',
      'user_pv_brand_list',
      'user_cart_brand_list',
      'user_fav_brand_list',
      'user_buy_brand_list'])
  parser.add_argument(
    '--non-user-cols', nargs='+', default=[
      'label',
      'ts',
      'item_price',
      'pid',
      'ad',
      'ad_campaign',
      'ad_customer',
      'item_category',
      'item_brand'])
  parser.add_argument('fname')
  parser.add_argument('output_fname')
  main(parser.parse_args())

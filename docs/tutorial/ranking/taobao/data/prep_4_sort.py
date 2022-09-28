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

Step 4: Sort click logs by user to reduce size.

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
import tqdm


def main(args):
  dtypes = {
    'ts': np.int64,
    'pid': np.int32,
    'user': np.int32,
    'user_cms_seg': np.int32,
    'user_cms_group': np.int32,
    'user_gender': np.int32,
    'user_age': np.int32,
    'user_pvalue': np.int32,
    'user_shopping': np.int32,
    'user_occupation': np.int32,
    'user_city': np.int32,
    'label': np.int32,
    'ad': np.int32,
    'ad_campaign': np.int32,
    'ad_customer': np.int32,
    'item_category': np.int32,
    'item_brand': np.int32,
    'item_price': np.float32}
  columns = [
    'ts',
    'pid',
    'label',
    'ad',
    'ad_campaign',
    'ad_customer',
    'item_category',
    'item_brand',
    'item_price',
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
    'user_pv_brand_list',
    'user_cart_category_list',
    'user_cart_brand_list',
    'user_fav_category_list',
    'user_fav_brand_list',
    'user_buy_category_list',
    'user_buy_brand_list']
  pa_schema = pa.schema([
    ('ts', pa.int64()),
    ('pid', pa.int32()),
    ('label', pa.int32()),
    ('ad', pa.int32()),
    ('ad_campaign', pa.int32()),
    ('ad_customer', pa.int32()),
    ('item_category', pa.int32()),
    ('item_brand', pa.int32()),
    ('item_price', pa.float32()),
    ('user', pa.int32()),
    ('user_cms_seg', pa.int32()),
    ('user_cms_group', pa.int32()),
    ('user_gender', pa.int32()),
    ('user_age', pa.int32()),
    ('user_pvalue', pa.int32()),
    ('user_shopping', pa.int32()),
    ('user_occupation', pa.int32()),
    ('user_city', pa.int32()),
    ('user_pv_category_list', pa.list_(pa.int32())),
    ('user_pv_brand_list', pa.list_(pa.int32())),
    ('user_cart_category_list', pa.list_(pa.int32())),
    ('user_cart_brand_list', pa.list_(pa.int32())),
    ('user_fav_category_list', pa.list_(pa.int32())),
    ('user_fav_brand_list', pa.list_(pa.int32())),
    ('user_buy_category_list', pa.list_(pa.int32())),
    ('user_buy_brand_list', pa.list_(pa.int32()))])
  logging.info('Sorting click logs...')
  for day in tqdm.tqdm(range(args.ndays)):
    fname = args.fname_template.format(day)
    output_fname = args.output_fname_template.format(day)
    click_log = pd.read_parquet(fname)
    click_log = click_log.astype(dtypes)
    click_log = click_log.set_index(['ts', 'pid', 'user', 'ad'])
    click_log = click_log.sort_index()
    click_log = click_log.reset_index()
    click_log = click_log[columns]
    click_log_table = pa.Table.from_pandas(click_log, preserve_index=False)
    click_log_table = click_log_table.cast(pa_schema)
    pq.write_table(
      click_log_table, output_fname,
      row_group_size=args.row_group_size,
      use_dictionary=not args.no_use_dictionary,
      compression=args.compression,
      flavor=args.flavor)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--row_group_size', type=int, default=1000000)
  parser.add_argument('--compression', default='zstd')
  parser.add_argument('--flavor', default='spark')
  parser.add_argument('--no-use-dictionary', default=False, action='store_true')
  parser.add_argument('--ndays', type=int, default=8)
  parser.add_argument('--fname-template', default='./merged_day_{}.parquet')
  parser.add_argument('--output-fname-template', default='./day_{}.parquet')
  main(parser.parse_args())

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

Step 1: Build backbone click logs.

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


def main(args):
  def fillnull(i):
    if not i:
      return args.null_value
    return i

  logging.info('Reading raw click logs from %s...', args.raw_sample_fname)
  raw_sample_dtypes = {
    'clk': np.int32,
    'pid': str,
    'time_stamp': np.int64,
    'user': np.int32,
    'adgroup_id': np.int32
  }
  raw_sample_columns = ['clk', 'pid', 'time_stamp', 'user', 'adgroup_id']
  raw_sample = pd.read_csv(
    args.raw_sample_fname,
    sep=',',
    usecols=raw_sample_columns,
    dtype=raw_sample_dtypes).reindex(columns=raw_sample_columns)
  raw_sample = raw_sample.rename(
    columns={'time_stamp': 'ts', 'adgroup_id': 'ad', 'clk': 'label'})
  raw_sample['pid'] = raw_sample['pid'].apply(
    lambda x: int(x.replace('_', ''))).astype(np.int32)
  click_log = raw_sample

  logging.info('Reading ad features from %s...', args.ad_feature_fname)
  ad_feature_converters = {
    'adgroup_id': np.int32,
    'cate_id': fillnull,
    'campaign_id': fillnull,
    'customer': fillnull,
    'brand': lambda i: int(i) if i and i != 'NULL' else args.null_value,
    'price': fillnull,
  }
  ad_feature_dtypes = {
    'adgroup_id': np.int32,
    'cate_id': np.int32,
    'campaign_id': np.int32,
    'customer': np.int32,
    'brand': np.int32,
    'price': np.float32
  }
  ad_feature = pd.read_csv(
    args.ad_feature_fname,
    sep=',',
    converters=ad_feature_converters)
  ad_feature = ad_feature.astype(ad_feature_dtypes)
  ad_feature = ad_feature.rename(
    columns={
      'adgroup_id': 'ad',
      'campaign_id': 'ad_campaign',
      'customer': 'ad_customer',
      'cate_id': 'item_category',
      'brand': 'item_brand',
      'price': 'item_price'})

  logging.info('Merging Ad features...')
  click_log = click_log.merge(
    ad_feature, on='ad', how='left')
  del raw_sample
  del ad_feature

  logging.info('Reading user profiles from %s...', args.user_profile_fname)
  user_profile_converters = {
    'userid': np.int32,
    'cms_segid': fillnull,
    'cms_group_id': fillnull,
    'final_gender_code': fillnull,
    'age_level': fillnull,
    'pvalue_level': fillnull,
    'shopping_level': fillnull,
    'occupation': fillnull,
    'new_user_class_level ': fillnull,
  }
  user_profile_dtypes = {
    'userid': np.int32,
    'cms_segid': np.int32,
    'cms_group_id': np.int32,
    'final_gender_code': np.int32,
    'age_level': np.int32,
    'pvalue_level': np.int32,
    'shopping_level': np.int32,
    'occupation': np.int32,
    'new_user_class_level ': np.int32
  }
  user_profile = pd.read_csv(
    args.user_profile_fname,
    sep=',',
    converters=user_profile_converters).astype(user_profile_dtypes)
  user_profile = user_profile.rename(
    columns={
      'userid': 'user',
      'cms_segid': 'user_cms_seg',
      'cms_group_id': 'user_cms_group',
      'final_gender_code': 'user_gender',
      'age_level': 'user_age',
      'occupation': 'user_occupation',
      'pvalue_level': 'user_pvalue',
      'shopping_level': 'user_shopping',
      'new_user_class_level ': 'user_city'})

  logging.info('Merging user profiles...')
  click_log = click_log.merge(user_profile, on='user', how='left')
  del user_profile

  logging.info('Spliting click logs...')
  click_log = click_log.fillna(args.null_value)
  click_log_dtypes = {
    'user_cms_seg': np.int32,
    'user_cms_group': np.int32,
    'user_gender': np.int32,
    'user_age': np.int32,
    'user_occupation': np.int32,
    'user_pvalue': np.int32,
    'user_shopping': np.int32,
    'user_city': np.int32
  }
  click_log = click_log.astype(click_log_dtypes)
  click_log.sort_values(['ts'])
  click_log['day'] = (click_log.ts - args.begin_ts) // 86400
  groups = click_log.groupby('day', as_index=False, sort=False)

  for day in range(args.ndays):
    output_fname = args.output_fname_template.format(day)
    logging.info('Writing parquet file %s...', output_fname)
    ddf = groups.get_group(day)
    del ddf['day']
    ddf_table = pa.Table.from_pandas(ddf, preserve_index=False)
    pq.write_table(
      ddf_table, output_fname,
      use_dictionary=not args.no_use_dictionary,
      compression=args.compression,
      flavor=args.flavor)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--compression', default='zstd')
  parser.add_argument('--flavor', default='spark')
  parser.add_argument('--no-use-dictionary', default=False, action='store_true')
  parser.add_argument('--null-value', type=int, default=-1 << 16)
  parser.add_argument('--begin-ts', type=int, default=1494000000)
  parser.add_argument('--ndays', type=int, default=8)
  parser.add_argument('--raw_sample_fname', default='./raw_sample.csv')
  parser.add_argument('--user_profile_fname', default='./user_profile.csv')
  parser.add_argument('--ad_feature_fname', default='./ad_feature.csv')
  parser.add_argument(
    '--output-fname-template', default='./backbone_day_{}.parquet')
  main(parser.parse_args())

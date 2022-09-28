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

r'''Convert Taobao Click Logs Dataset to TFRecord.

See https://tianchi.aliyun.com/dataset/dataDetail?dataId=56 for more
information.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import pandas as pd
import tensorflow as tf
import tqdm


def main(args):
  df = pd.read_parquet(args.fname)
  options = tf.io.TFRecordOptions(
    compression_type=tf.io.TFRecordCompressionType.GZIP)
  with tf.io.TFRecordWriter(args.output_fname, options) as writer:
    for row in tqdm.tqdm(df.itertuples(), total=len(df)):
      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'ts': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.ts])),
            'pid': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.pid])),
            'label': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.label])),
            'ad': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.ad])),
            'ad_campaign': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.ad_campaign])),
            'ad_customer': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.ad_customer])),
            'item_category': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.item_category])),
            'item_brand': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.item_brand])),
            'item_price': tf.train.Feature(
              float_list=tf.train.FloatList(value=[row.item_price])),
            'user': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.user])),
            'user_cms_seg': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.user_cms_seg])),
            'user_cms_group': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.user_cms_group])),
            'user_gender': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.user_gender])),
            'user_age': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.user_age])),
            'user_pvalue': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.user_pvalue])),
            'user_shopping': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.user_shopping])),
            'user_occupation': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.user_occupation])),
            'user_city': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[row.user_city])),
            'user_pv_category_list': tf.train.Feature(
              int64_list=tf.train.Int64List(value=row.user_pv_category_list)),
            'user_pv_brand_list': tf.train.Feature(
              int64_list=tf.train.Int64List(value=row.user_pv_brand_list)),
            'user_cart_category_list': tf.train.Feature(
              int64_list=tf.train.Int64List(value=row.user_cart_category_list)),
            'user_cart_brand_list': tf.train.Feature(
              int64_list=tf.train.Int64List(value=row.user_cart_brand_list)),
            'user_fav_category_list': tf.train.Feature(
              int64_list=tf.train.Int64List(value=row.user_fav_category_list)),
            'user_fav_brand_list': tf.train.Feature(
              int64_list=tf.train.Int64List(value=row.user_fav_brand_list)),
            'user_buy_category_list': tf.train.Feature(
              int64_list=tf.train.Int64List(value=row.user_buy_category_list)),
            'user_buy_brand_list': tf.train.Feature(
              int64_list=tf.train.Int64List(value=row.user_buy_brand_list))}))
      writer.write(example.SerializeToString())


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('fname')
  parser.add_argument('output_fname')
  main(parser.parse_args())

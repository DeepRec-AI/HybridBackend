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

r'''Calculate statistics of Taobao Click Logs Dataset.

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
import tqdm


def main(args):
  users = []
  ads = []
  categories = []
  brands = []
  for day in tqdm.tqdm(range(args.ndays)):
    fname = args.fname_template.format(day)
    click_log = pd.read_parquet(fname)
    users += pd.unique(click_log['user']).tolist()
    ads += pd.unique(click_log['ad']).tolist()
    categories += pd.unique(click_log['item_category']).tolist()
    brands += pd.unique(click_log['item_brand']).tolist()
    del click_log
  users = np.unique(users)
  logging.info('#users = %d', len(users))
  del users
  ads = np.unique(ads)
  logging.info('#ads = %d', len(ads))
  del ads
  categories = np.unique(categories)
  logging.info('#categories = %d', len(categories))
  del categories
  brands = np.unique(brands)
  logging.info('#brands = %d', len(brands))
  del brands


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--ndays', type=int, default=8)
  parser.add_argument('--fname-template', default='./day_{}.parquet')
  main(parser.parse_args())

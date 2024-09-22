# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/27 20:50
@Auth ： xiaolongtuan
@File ：load_pkl.py
"""
import pickle
from bgp_semantics import BgpSemantics
import pandas as pd
# 打开文件并加载内容
def load_pd(path):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)  # 设置显示宽度，以防止换行
    df = pd.read_json(path, lines=True)

    print(df.loc[df['changed'] == 1].head(10))

def load_pkl(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
        print(loaded_data)

load_pkl('data/0/9_state.pkl')
# load_pd('ospf_update_dataset/s/0/dataset.jsonl')

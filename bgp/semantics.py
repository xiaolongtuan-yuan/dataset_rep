'''
模拟网络转发行为获取数据集
'''

import multiprocessing
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import torch
from multiprocessing import Pool
import os
import shutil

class Semantics:
    def __init__(self):
        pass

    def sample(self, seed=None):
        raise NotImplementedError()

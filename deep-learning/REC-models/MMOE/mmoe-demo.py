# 实现mmoe代码
# 数据集：https://archive-beta.ics.uci.edu/ml/datasets/census+income+kdd
# 预测收入、婚姻状态两个目标
# 论文原文：https://dl.acm.org/doi/pdf/10.1145/3219819.3220007

import numpy as np
import random
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

random.seed(24)
np.random.seed(24)
seed = 24
batch_size = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1]==1 and len(input_shape)>1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel() # https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes, )
    categorical = np.reshape(categorical, output_shape)
    return categorical


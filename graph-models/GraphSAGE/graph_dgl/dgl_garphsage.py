# 参考：https://zhuanlan.zhihu.com/p/344067462

import torch as torch
import dgl
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 定义图神经网络GraphSAGE
from dgl.nn.pytorch import SAGEConv


class myGraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator):
        super(myGraphSAGE, self).__init__()
        self.n_layers = n_layers
# 参考：https://zhuanlan.zhihu.com/p/344067462
# https://zhuanlan.zhihu.com/p/348382635

import torch as torch
import dgl
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 定义图神经网络GraphSAGE
from dgl.nn.pytorch import SAGEConv


class MyGraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator):
        super(MyGraphSAGE, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layer = nn.ModuleList()
        self.layer.append(SAGEConv(in_feats, n_hidden, aggregator))
        for i in range(1, n_layers - 1):
            self.layer.append(SAGEConv(n_hidden, n_hidden, aggregator))
        self.layer.append(SAGEConv(n_hidden, n_classes, aggregator))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, feats):
        """
        前向计算函数
        blocks:
        feats:
        """
        h = feats # 应该是最初始输入的节点特征
        for i, (layer, block) in enumerate(zip(self.layer, blocks)):
            h = layer(block, h)
            if i != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


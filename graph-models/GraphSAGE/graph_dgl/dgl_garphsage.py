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


# tools
def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def evaluate(model, my_net, labels, val_nid, val_mask, batch_s, num_worker, device):
    model.eval()
    with torch.no_grad():
        label_pred = model.inference(my_net, val_nid, batch_s, num_worker, device)
    model.train()
    return (torch.argmax(label_pred[val_mask], dim=1) == labels[val_mask]).float().sum() / len(label_pred[val_mask])


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

        # SAGEConv: https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.SAGEConv.html
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
    
    def inference(self, my_net, val_nid, batch_s, num_worker, device):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        dataloader = dgl.dataloading.NodeDataLoader(
            my_net,
            val_nid,
            sampler,
            batch_size=batch_s,
            shuffle=True,
            drop_last=False,
            num_workers=num_worker
        )
        ret = torch.zeros(my_net.num_nodes(), self.n_classes)
        
        for input_nodes, output_nodes, blocks in dataloader:
            h = blocks[0].srcdata['feature'].to(device)
            for i, (layer, block) in enumerate(zip(self.layer, blocks)):
                block = block.int().to(device)
                h = layer(block, h)
                if i != self.n_layers - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
            ret[output_nodes] = h.cpu()
        return ret 
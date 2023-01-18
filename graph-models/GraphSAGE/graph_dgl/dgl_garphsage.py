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


# read data
def train_val_split(node_fea):
    # 划分数据集
    train_node_ids = np.array(node_fea.groupby('label_number')
                              .apply(lambda x: x.sort_values('node_id_number')['node_id_number'].values[:20]))
    val_node_ids = np.array(node_fea.groupby('label_number')
                            .apply(lambda x: x.sort_values('node_id_number')['node_id_number'].values[21:110]))
    test_node_ids = np.array(node_fea.groupby('label_number')
                             .apply(lambda x: x.sort_values('node_id_number')['node_id_number'].values[111:300]))

    train_nid = []
    val_nid = []
    test_nid = []
    for (train_nodes, val_nodes, test_nodes) in zip(train_node_ids, val_node_ids, test_node_ids):
        train_nid.append(train_nodes)
        val_nid.append(val_nodes)
        test_nid.append(test_nodes)

    train_mask = node_fea['node_id_number'].apply(lambda x: x in train_nid)
    val_mask = node_fea['node_id_number'].apply(lambda x: x in val_nid)
    test_mask = node_fea['node_id_number'].apply(lambda x: x in test_nid)
    return train_mask, test_mask, val_mask, train_nid, test_nid, val_nid


def load_data():
    node_fea = pd.read_table('../cora/cora.content', header=None)
    edges = pd.read_table('../cora/cora.cites', header=None)
    # 0是node id， 1434是node label
    node_fea.rename(columns={0:'node_id', 1434:'label'}, inplace=True)

    nodeid_number_dict = dict(zip(node_fea['node_id'].unique(),
                                  range(node_fea['node_id'].nunique())))
    node_fea['node_id_number'] = node_fea['node_id'].map(nodeid_number_dict)
    edges['edge1'] = edges[0].map(nodeid_number_dict)
    edges['edge2'] = edges[1].map(nodeid_number_dict)

    label_dict = dict(zip(node_fea['label'].unique(),
                          range(node_fea['label'].nunique())))
    node_fea['label_number'] = node_fea['label'].map(label_dict)

    src = np.array(edges['edge1'].values)
    dst = np.array(edges['edge2'].values)

    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])

    my_net = dgl.DGLGraph((u,v))

    fea_id = range(1, 1434)
    tensor_fea = torch.tensor(node_fea[fea_id].values, dtype=torch.float32)

    fea_np = nn.Embedding(2708, 1433)
    fea_np.weight = nn.Parameter(tensor_fea)

    my_net.ndata['features'] = fea_np.weight
    my_net.ndata['label'] = torch.tensor(node_fea['label_number'].values)

    in_feats = 1433
    n_classes = node_fea['label'].nunique()

    data = in_feats, n_classes, my_net, fea_np
    train_val_data = train_val_split(node_fea)
    return data, train_val_data


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
        h = feats  # 应该是最初始输入的节点特征
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

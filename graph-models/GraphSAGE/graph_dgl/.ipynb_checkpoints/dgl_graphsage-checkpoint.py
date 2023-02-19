# 参考：https://zhuanlan.zhihu.com/p/344067462
# https://zhuanlan.zhihu.com/p/348382635

import torch as torch
import dgl
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import itertools

# 定义图神经网络GraphSAGE
from dgl.nn.pytorch import SAGEConv


# read data
def train_val_split(node_fea):
    # 划分数据集
    train_node_ids = np.array(node_fea.groupby('label_number')
                              .apply(lambda x: x.sort_values('node_id_number')['node_id_number'].values[:100]))
    val_node_ids = np.array(node_fea.groupby('label_number')
                            .apply(lambda x: x.sort_values('node_id_number')['node_id_number'].values[100:200]))
    test_node_ids = np.array(node_fea.groupby('label_number')
                             .apply(lambda x: x.sort_values('node_id_number')['node_id_number'].values[200:300]))

    train_nid = []
    val_nid = []
    test_nid = []
    for (train_nodes, val_nodes, test_nodes) in zip(train_node_ids, val_node_ids, test_node_ids):
        train_nid.extend(train_nodes)
        val_nid.extend(val_nodes)
        test_nid.extend(test_nodes)

    train_mask = node_fea['node_id_number'].apply(lambda x: x in train_nid)
    val_mask = node_fea['node_id_number'].apply(lambda x: x in val_nid)
    test_mask = node_fea['node_id_number'].apply(lambda x: x in test_nid)
    return train_mask, test_mask, val_mask, train_nid, test_nid, val_nid


def load_data():
    node_fea = pd.read_table('../cora/cora.content', header=None)
    edges = pd.read_table('../cora/cora.cites', header=None)
    # 0是node id， 1434是node label
    node_fea.rename(columns={0: 'node_id', 1434: 'label'}, inplace=True)

    node_id_number_dict = dict(zip(node_fea['node_id'].unique(),
                                   range(node_fea['node_id'].nunique())))
    node_fea['node_id_number'] = node_fea['node_id'].map(node_id_number_dict)
    edges['edge1'] = edges[0].map(node_id_number_dict)
    edges['edge2'] = edges[1].map(node_id_number_dict)

    label_dict = dict(zip(node_fea['label'].unique(),
                          range(node_fea['label'].nunique())))
    node_fea['label_number'] = node_fea['label'].map(label_dict)

    src = np.array(edges['edge1'].values)
    dst = np.array(edges['edge2'].values)

    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])

    my_net = dgl.DGLGraph((u, v))

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
        """
        in_feats: 输入的特征数量
        n_hidden: 内部的隐藏层的维度
        n_classes: 分类数目
        n_layers: 层数
        activation: 激活函数
        dropout: dropout比例
        aggregator: 聚合方法 Aggregator type to use (mean, gcn, pool, lstm)
        """
        super(MyGraphSAGE, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layer = nn.ModuleList()  # layer 是一连串的处理手段

        # SAGEConv: https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.SAGEConv.html
        # 第一层是输入的特征，映射到隐藏层
        self.layer.append(SAGEConv(in_feats, n_hidden, aggregator))
        # 第二到倒数第二层，输入和输出都是 隐藏层的维度
        for i in range(1, n_layers - 1):
            self.layer.append(SAGEConv(n_hidden, n_hidden, aggregator))
        # 最后一层，是隐藏层到类别的映射
        self.layer.append(SAGEConv(n_hidden, n_classes, aggregator))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, feats):
        """
        前向计算函数
        blocks: 待理解
        feats: 最初的特征输入
        layer: 是SAGEConv的每一层，block和h都是其输入特征
        """
        h = feats  # 应该是最初始输入的节点特征
        for i, (layer, block) in enumerate(zip(self.layer, blocks)):
            h = layer(block, h)
            if i != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, my_net, val_nid, batch_s, num_worker, device):
        """
        my_net:
        val_nid:
        batch_s:
        """
        # 采样类
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
        # 最后返回结果： 节点数，节点类别
        ret = torch.zeros(my_net.num_nodes(), self.n_classes)

        # 推理计算过程
        # dataloader 返回 输入节点，输出节点，blocks
        for input_nodes, output_nodes, blocks in dataloader:
            h = blocks[0].srcdata['features'].to(device)
            for i, (layer, block) in enumerate(zip(self.layer, blocks)):
                block = block.int().to(device)
                h = layer(block, h)
                if i != self.n_layers - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
            ret[output_nodes] = h.cpu()
        return ret


# tools
def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def evaluate(model, my_net, labels, val_nid, val_mask, batch_s, num_worker, device):
    """
    评估函数，对于val_mask的节点判断准确率
    """
    model.eval()
    with torch.no_grad():
        label_pred = model.inference(my_net, val_nid, batch_s, num_worker, device)
    model.train()
    return (torch.argmax(label_pred[val_mask], dim=1) == labels[val_mask]).float().sum() / len(label_pred[val_mask])


#  调用过程
def run(data, train_val_data, args, sample_size, learning_rate, device_num):
    if device_num > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    in_feats, n_classes, my_net, fea_para = data
    hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_workers = args
    train_mask, test_mask, val_mask, train_nid, test_nid, val_nid = train_val_data

    n_feat = my_net.ndata['features']
    labels = my_net.ndata['label']
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_size)  # sample_size = [5,10] 表示在第一层抽样5个邻居，第二层抽样10个邻居
    dataloader = dgl.dataloading.NodeDataLoader(
        my_net,
        train_nid,
        sampler,
        batch_size=batch_s,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )
    model = MyGraphSAGE(in_feats,
                        hidden_size,
                        n_classes,
                        n_layers,
                        activation,
                        dropout,
                        aggregator)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    for epoch in range(100):
        for batch, (input_nodes, output_nodes, block) in enumerate(dataloader):
            # block: [Block(num_src_nodes=1348, num_dst_nodes=565, num_edges=2723), Block(num_src_nodes=565,
            # num_dst_nodes=128, num_edges=646)] 是list
            batch_feature, batch_label = load_subtensor(n_feat, labels, output_nodes, input_nodes, device)
            block = [block_.int().to(device) for block_ in block]
            model_pred = model(block, batch_feature)
            loss = loss_fn(model_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                print('Batch %d | loss: %.4f' % (batch, loss.item()))

            if epoch % 10 == 0:
                print('----')
                val_acc = evaluate(model, my_net, labels, val_nid, val_mask,
                                   batch_s, num_workers, device)
                train_acc = evaluate(model, my_net, labels, train_nid, train_mask,
                                     batch_s, num_workers, device)
                print('Epoch %d val acc: %.4f, train acc: %.4f' % (epoch, val_acc.item(), train_acc.item()))

    acc_test = evaluate(model, my_net, labels, test_nid, test_mask,
                        batch_s, num_workers, device)
    print('Test acc: %.4f' % (acc_test.item()))
    return model


data, train_val_data = load_data()
hidden_size = 120
n_layers = 2
sample_size = [10, 25]
activation = F.relu
dropout = 0.5
aggregator = 'mean'
batch_s = 128
num_worker = 0
learning_rate = 0.003
device_num = 0

args = hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker
trained_model = run(data, train_val_data, args, sample_size, learning_rate, device_num)

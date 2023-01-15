import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from encoders import Encoder
from aggregators import MeanAggregator


# python 3.7
# torch 1.10.1


def load_cora():
    """
    return:
        feat_data: shape(num_nodes, num_feats), the original features of each node
        labels: shape(num_nodes, 1), the label of each node
        adj_lists: a dict of set, key = node, set = the nodes it links to, as a set
    """
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("../cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    edge_lists = []
    with open("../cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2) # 在此转化为 0～2707
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=True)

    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists,
                   agg2, base_model=enc1, gcn=True, cuda=False)
    enc1.num_sample = 5
    enc2.num_sample = 5

    myGraphsage = SupervisedGraphSage(7, enc2)
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, myGraphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256] # list of 256 node num
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = myGraphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        # print(batch, loss.data[0])

    val_output = myGraphsage.forward(val) # tensor, shape(500, 7)
    print("Validation F1: ", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average='micro'))
    print("Average batch time: ", np.mean(times))
    return features


if __name__ == "__main__":
    trained_features = run_cora()
    print(trained_features(torch.tensor(0)))

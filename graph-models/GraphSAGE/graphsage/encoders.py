import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, aggregator,
                 num_sample=10,
                 base_model=None, gcn=False, cuda=False,
                 feature_transform=False):
        super(Encoder, self).__init__()
        if base_model is not None:
            self.base_model = base_model

        self.features = features
        self.feature_dim = feature_dim
        self.adj_lists = adj_lists  # 连接
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feature_dim if self.gcn else 2 * self.feature_dim)
        )
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        generates embeddings for a batch of nodes
        nodes: a list of nodes
        """
        neigh_features = self.aggregator.forward(nodes,
                                                 [self.adj_lists[int(node)] for node in nodes],
                                                 self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_features = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_features = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_features, neigh_features], dim=1)
        else:
            combined = neigh_features
        combined = F.relu(self.weight.mm(combined.t()))
        return combined

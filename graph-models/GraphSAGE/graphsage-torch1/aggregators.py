import torch
import torch.nn as nn
from torch.autograd import Variable
import random


class MeanAggregator(nn.Module):
    """
    Aggregate a node's embeddings using mean of neighbors' embeddings.
    """

    def __init__(self, features, cuda=False, gcn=False):
        """

        """
        super(MeanAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample = 10):
        """
        nodes: a list of nodes in a batch, shape=[batch,1]
        to_neighs: list of set, each set is the set of neighbors for the node in a batch, shape=[batch,set()]
        num_sample: number of neighbors to sample, so to control the complexity of one batch calculation
        """
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample))
                           if len(to_neigh) >= num_sample else to_neigh
                           for to_neigh in to_neighs] # 返回list，to_neighs 里每个set，仅抽样num_sample个邻居节点
        else:
            samp_neighs = to_neighs # 没指定时，全抽样

        # gcn时，圈定本节点，以及邻居节点
        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]])
                           for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i, n in enumerate(unique_nodes_list)} # 全部nodes的临时编号
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n]
                          for samp_neigh in samp_neighs
                          for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats
import sys
import os
from collections import defaultdict
import numpy as np


def _split_data(num_nodes, test_split=3, val_split=6):
    rand_indices = np.random.permutation(num_nodes)

    test_size = num_nodes // test_split
    val_size = num_nodes // val_split
    train_size = num_nodes - test_size - val_size

    test_indexes = rand_indices[:test_size]
    val_indexes = rand_indices[test_size:(test_size + val_size)]
    train_indexes = rand_indices[(test_size + val_size):]

    return test_indexes, val_indexes, train_indexes


class DataCenter:

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def load_dataset(self, dataset='cora'):
        if dataset == 'cora':
            cora_content_file = self.config['file_path.cora_content']
            cora_cite_file = self.config['file_path.cora_cite']

            feat_data = []
            labels = []
            node_map = {}  # 原node节点 j to 归一化 i
            label_map = {}
            with open(cora_content_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    feat_data.append([float(x) for x in info[1:-1]])  # 1403 位特征
                    node_map[info[0]] = i
                    if not info[-1] in label_map:
                        label_map[info[-1]] = len(label_map)
                    labels.append(label_map[info[-1]])
            feat_data = np.asarray(feat_data)
            labels = np.asarray(labels, dtype=np.int64)

            adj_lists = defaultdict(set)
            with open(cora_cite_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    assert len(info) == 2
                    paper1 = node_map[info[0]]
                    paper2 = node_map[info[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)

            assert len(feat_data) == len(labels) == len(adj_lists)  # 但有全部节点不一定有连接起来的节点
            test_indexes, val_indexes, train_indexes = _split_data(feat_data.shape[0])

            setattr(self, dataset + '_test', test_indexes)
            setattr(self, dataset + '_val', val_indexes)
            setattr(self, dataset + '_train', train_indexes)

            setattr(self, dataset + '_feats', feat_data)
            setattr(self, dataset + '_labels', labels)
            setattr(self, dataset + '_adj_lists', adj_lists)

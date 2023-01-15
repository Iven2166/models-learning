import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid
# from torch_geometric.loader import LinkNeighborLoader
# from torch_geometric.nn import GraphSAGE


print(osp.dirname(osp.realpath(__file__)))

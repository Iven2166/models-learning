import sys
import os
import torch
import argparse
import pyhocon
import random

from src.dataCenter import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=20)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='./src/experiments.conf')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
    # load config
    config = pyhocon.ConfigFactory.parse_file(args.config)
    print(config)
    # load data
    ds = args.dataset
    datacenter = DataCenter(config)
    datacenter.load_dataset(ds)
    features = torch.FloatTensor(getattr(datacenter, ds + '_feats')).to(device)
    num_labels = len(set(getattr(datacenter, ds + '_labels')))
    print(features, features.shape)
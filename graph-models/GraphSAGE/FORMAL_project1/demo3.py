import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp
import dgl
import torch
from dgl.data import AsNodePredDataset#, AsLinkPredDataset
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
from dgl.nn import SAGEConv
import dgl.function as fn
import tqdm
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import os
import logging

def load_cora(data_path):
    data0 = dgl.data.CSVDataset(data_path,force_reload=True)
    data = AsNodePredDataset(data0, split_ratio=(0.5,0.2,0.3))
    g = data[0]
    g.ndata["features"] = g.ndata.pop("feat").float()
    g.ndata["labels"] = g.ndata.pop("label")
    return g, data.num_classes


# 制定model
class Model(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(Model, self).__init__()
        self.h_feats = h_feats
        self.in_feats = in_feats
        self.conv1 = SAGEConv(self.in_feats, self.h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(self.h_feats, self.h_feats, aggregator_type='mean')
        
    def forward(self, mfgs, x):
        h_dst = x[: mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[: mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h
    

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # return g.edata['score'][:,0]
            return torch.sigmoid(g.edata['score'][:,0]) # 相比原版接了sigmoid缩放到0、1区间
        

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges_method(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges_method)
            return g.edata['score'][:,0]


# 评估模块
def compute_auc(pos_score, neg_score):
    with torch.no_grad():
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        return roc_auc_score(labels, scores)
    
    
def get_test_result(model, predictor, test_dataloader, threds):
    model.eval()
    with torch.no_grad(), test_dataloader.enable_cpu_affinity():
        pred_all, pred01_all, label_all = torch.Tensor(), torch.Tensor(), torch.Tensor()
        loss_all = 0
        for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(test_dataloader):
            pos_graph = pos_graph.to(gpu_device)
            neg_graph = neg_graph.to(gpu_device)
            mfgs = [mfg.int().to(gpu_device) for mfg in mfgs]

            inputs = mfgs[0].srcdata['features']
            outputs = model(mfgs, inputs)
            pos_score = predictor(pos_graph, outputs)
            neg_score = predictor(neg_graph, outputs)

            # the score and label of edges (real and non-existent)
            score = torch.cat([pos_score, neg_score])
            label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            loss = F.binary_cross_entropy_with_logits(score, label)

            pred = score.cpu()
            pred01 = pred.detach()
            pred01[pred01>=threds] = 1
            pred01[pred01<threds] = 0
            
            pred_all = torch.cat([pred_all, pred])
            pred01_all = torch.cat([pred01_all, pred01])
            label_all = torch.cat([label_all, label.cpu()])
            loss_all += loss
            
        accu = sklearn.metrics.accuracy_score(label_all.cpu().numpy(),pred01_all)
        auc = roc_auc_score(label_all.cpu().numpy(), pred_all)
        size = len(test_dataloader)
            # tq.set_postfix({'test-loss': '%.03f' % loss.item(), 
            #                 'test-accu': '%0.3f'%accu.item(),
            #                 'test-auc': '%0.3f'%auc.item()
            #                }, refresh=False)
    return accu, auc, pred_all, pred01_all, label_all, loss_all, size

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # Output to file
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Output to terminal
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


if __name__ == "__main__":
    gpu_device = torch.device('cuda')
    cpu_device = torch.device('cpu')
    print(gpu_device, cpu_device)
    data_path = '../../dataset/cora_csv_idrank/'
    raw_g, n_classes = load_cora(data_path)
    g = dgl.add_reverse_edges(raw_g)
    node_features = g.ndata['features']
    node_labels = g.ndata['labels']
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()
    print('Number of classes: {:d}'.format(num_classes))

    # reverse id 的规定
    """
    reverse_eids (Tensor or dict[etype, Tensor], optional) –
    A tensor of reverse edge ID mapping. The i-th element indicates the ID of the i-th edge’s reverse edge.
    """
    E = g.number_of_edges()
    reverse_eids = torch.cat([torch.arange(E//2,  E), torch.arange(0, E//2)])
    reverse_eids

    # 规定提取的边
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)

    train_size = int(0.7 * len(eids))
    train_eid = eids[:train_size]
    # train_g = dgl.remove_edges(g, eids[train_size:])

    test_eid = eids[train_size:]
    # test_g = dgl.remove_edges(g, eids[:train_size])

    # 建立样本迭代器-base
    # https://docs.dgl.ai/generated/dgl.dataloading.DataLoader.html
    # https://docs.dgl.ai/en/0.8.x/generated/dgl.dataloading.as_edge_prediction_sampler.html#dgl.dataloading.as_edge_prediction_sampler

    negative_sampler = dgl.dataloading.negative_sampler.Uniform(3)  # N = 3
    neighbor_sampler = dgl.dataloading.MultiLayerNeighborSampler([10,5])
    edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler=neighbor_sampler,
        negative_sampler=negative_sampler
    )
    edge_sampler_excl = dgl.dataloading.as_edge_prediction_sampler(
        sampler=neighbor_sampler,
        negative_sampler=negative_sampler,
        exclude='reverse_id',
        reverse_eids=reverse_eids
    )
    train_dataloader = dgl.dataloading.DataLoader(
        graph = g, 
        indices = train_eid, # indices=torch.arange(g.number_of_edges()), 
        graph_sampler=edge_sampler_excl,
        batch_size=512,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        device='cpu'
        # use_prefetch_thread=True,
        # pin_prefetcher=True, 
        # use_ddp=True       # Make it work with distributed data parallel
    )

    test_dataloader = dgl.dataloading.DataLoader(
        graph=g,
        indices = test_eid, #indices=torch.arange(g.number_of_edges()),
        graph_sampler=edge_sampler,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        device='cpu'
    )

    # 实例化
    model = Model(num_features, 256).to(gpu_device)
    predictor = DotPredictor().to(gpu_device)
    opt = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()))

    # 训练过程
    best_test_auc = 0
    best_model_path = './demo3-models/'

    logger = get_logger('./demo3-bash-train.log')
    logger.info('start training!')
    EPOCHS = 50
    for epoch in range(EPOCHS):
        with train_dataloader.enable_cpu_affinity(), tqdm.tqdm(train_dataloader) as tq:
            # with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(tq):
                model.train()
                pos_graph = pos_graph.to(gpu_device)
                neg_graph = neg_graph.to(gpu_device)
                mfgs = [mfg.int().to(gpu_device) for mfg in mfgs]

                inputs = mfgs[0].srcdata['features']
                outputs = model(mfgs, inputs)
                pos_score = predictor(pos_graph, outputs)
                neg_score = predictor(neg_graph, outputs)

                # the score and label of edges (real and non-existent)
                score = torch.cat([pos_score, neg_score])
                label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                loss = F.binary_cross_entropy_with_logits(score, label)

                pred = score.cpu()
                pred01 = pred.detach()
                threds = 0.5
                pred01[pred01>=threds] = 1
                pred01[pred01<threds] = 0
                with torch.no_grad():
                    accu = sklearn.metrics.accuracy_score(label.cpu().numpy(),pred01)
                    auc = roc_auc_score(label.cpu().numpy(), score.detach().cpu())

                opt.zero_grad()
                loss.backward()
                opt.step()

                tq.set_postfix({
                    'train:':'-----',
                    'loss': '%.03f' % loss.item(), 
                    'accu': '%0.3f'%accu.item(),
                    'auc': '%0.3f'%auc.item()
                               }, refresh=False)
                logger.info('Epoch:[{}/{}]\t step:[{}] \t loss={:.5f}\t accu={:.3f} \t auc={:.3f}'.format(epoch, step, EPOCHS, loss, accu, auc))

            # train里的 每 N 个step执行一次
            if (epoch+1)%3==0:
                test_accu, test_auc, test_pred_all, test_pred01_all, test_label_all, test_loss_all, test_size = get_test_result(
                    model, predictor, test_dataloader, threds)
                print("------------ 执行测试集 ------------ :")
                logger.info('测试 Epoch:[{}/{}]\t loss={:.5f}\t accu={:.3f} auc={:.3f}'.format(epoch, EPOCHS, test_loss_all/test_size, test_accu, test_auc))
                print("test_loss={:.3f}, test_accu={:.3f}, test_auc={:.3f}".format(test_loss_all/test_size, test_accu, test_auc))
                if(best_test_auc < test_auc):
                    best_test_auc = test_auc
                    # model save
                    torch.save({'state':model.state_dict(), 'optimizer':opt.state_dict(), 'epoch':epoch, 'size':len(train_dataloader)}, 
                              os.path.join(best_model_path, 
                                           "model-demo3"+"batch_size="
                                           +str(len(train_dataloader))
                                           +"epoch="+str(epoch)
                                           +"test_auc="+str(round(test_auc,3))
                                           +"test_accu="+str(round(test_accu,3))
                                           +".pt"))
                

                # https://docs.dgl.ai/en/0.8.x/tutorials/blitz/4_link_predict.html 最终test预测到 auc=0.865 程度
    logger.info('finish training!')
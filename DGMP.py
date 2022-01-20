#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:00:47 2021

@author: xujingyu
"""
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3,0'

import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from DIGCNConv import DIGCNConv
from torch_scatter import scatter_add
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from torch.nn import Linear
from datasets import get_citation_dataset
from train_eval import run
import pickle as pkl
from torch_geometric.nn import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default='RegNetwork')
parser.add_argument('--gpu-no', type=int, default=0)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--recache', action="store_true", help="clean up the old adj data", default=True)
parser.add_argument('--normalize-features', action="store_true", default=False)
parser.add_argument('--adj-type', type=str, default='or')
parser.add_argument('--cv-runs', help='Number of cross validation runs',type=int,default=5)
args = parser.parse_args()

class InceptionBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InceptionBlock, self).__init__()
        self.ln = Linear(in_dim, out_dim)
        self.conv1 = GCNConv(in_dim, out_dim)
        self.conv2 = GCNConv(in_dim, out_dim)
        self.conv3 = GCNConv(in_dim, out_dim)
    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
    def forward(self, x, edge_index, edge_index1, edge_index2):
        x0 = self.ln(x)
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x, edge_index1)
        x3 = self.conv3(x, edge_index2)
        return x0, x1, x2, x3

class Sparse_Three_Concat(torch.nn.Module):
    def __init__(self, dataset):
        super(Sparse_Three_Concat, self).__init__()
        self.ib1 = InceptionBlock(dataset.num_features, args.hidden)
        self.ln1 = Linear(args.hidden* 4, dataset.num_classes)

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index1 = data.edge_index1
        edge_index2 = data.edge_index2
                
        x0,x1,x2,x3 = self.ib1(x, edge_index, edge_index1, edge_index2)
        x0 = F.dropout(x0, p=args.dropout, training=self.training)
        x1 = F.dropout(x1, p=args.dropout, training=self.training)
        x2 = F.dropout(x2, p=args.dropout, training=self.training)
        x3 = F.dropout(x3, p=args.dropout, training=self.training)
        
        x = torch.cat((x0,x1,x2,x3),1)
        x = self.ln1(x)
        
        return F.log_softmax(x, dim=1)

def run_IDGCN(dataset, gpu_no):
     cv_loss, cv_acc, cv_std, cv_time, output, data = [], [], [], [], [],[]
     for cv_run in range(args.cv_runs):
         print('cross validation for the {}th run'.format(cv_run+1))
         citation_dataset = get_citation_dataset(dataset, args.alpha, args.recache, args.normalize_features, 
                                   cv_run, args.adj_type)
         data.append(citation_dataset[0])
         
         # Replace Sparse_Three_Concat1 with Sparse_Three_Concat2 to test concat
         val_loss, test_acc, test_std, time, logits = run(citation_dataset, gpu_no, Sparse_Three_Concat(citation_dataset), 
                                 args.epochs, args.lr, args.weight_decay, args.early_stopping)
         cv_loss.append(val_loss)
         cv_acc.append(test_acc)
         cv_std.append(test_std)
         cv_time.append(time)
         output.append(logits)
        
     cv_loss = np.mean(cv_loss)
     cv_acc = np.mean(cv_acc)
     cv_std = np.mean(cv_std)
     cv_time = np.mean(cv_time)
        
     return cv_loss, cv_acc, cv_std, cv_time, output, data

if __name__ == '__main__':
    if args.dataset is not None:
        dataset_name = [args.dataset]
    else:
        dataset_name = ['cora_ml','citeseer']
    outputs = ['val_loss', 'test_acc', 'test_std', 'time']
    result = pd.DataFrame(np.arange(len(outputs)*len(dataset_name), dtype=np.float32).reshape(
             (len(dataset_name), len(outputs))), index=dataset_name, columns=outputs)

    for dataset in dataset_name:
        loss, acc, std, time, logits, data = run_IDGCN(dataset, args.gpu_no)
        result.loc[dataset]['loss_mean'] = loss
        result.loc[dataset]['acc_mean'] = acc
        result.loc[dataset]['std_mean'] = std
        result.loc[dataset]['time_mean'] = time
   
        
        with open('/home/disk1/xujingyu/桌面/gcn_data/first_revise/RegNetwork_DGMP.pkl', 'wb') as f:
            pkl.dump([logits, data, acc], f)
             
#nvidia-smi
#nvidia-smi -q

#htop

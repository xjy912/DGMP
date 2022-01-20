#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:23:06 2021

@author: xujingyu
"""
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.nn import Linear
from datasets import get_citation_dataset
from train_eval import run
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cancer1')
parser.add_argument('--gpu-no', type=int, default=0)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--recache', action="store_true",
                    help="clean up the old adj data", default=True)
parser.add_argument('--normalize-features', action="store_true", default=False)
parser.add_argument('--adj-type', type=str, default='or')
parser.add_argument('--cv-runs', help='Number of cross validation runs',type=int,
                    default=5
                    )
args = parser.parse_args()


class MLP(torch.nn.Module):
    def __init__(self,dataset):
        super(MLP, self).__init__()
        self.fc1 = Linear(dataset.num_features, args.hidden)
        self.fc2 = Linear(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, data): 
        x = data.x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
 
        return F.log_softmax(x, dim=1)
    
def cv_run_MLP(dataset, gpu_no):
     cv_loss, cv_acc, cv_std, cv_time, output, data = [], [], [], [], [],[]
     for cv_run in range(args.cv_runs):
         print('cross validation for the {} run'.format(cv_run+1))
         citation_dataset = get_citation_dataset(dataset, args.alpha, args.recache, args.normalize_features, 
                                   cv_run, args.adj_type)
         data.append(citation_dataset[0])
         val_loss, test_acc, test_std, time, logits = run(citation_dataset, gpu_no,MLP(citation_dataset), 
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
        #dataset_name = ['cora_ml','citeseer']
        dataset_name = ['citeseer']
    outputs = ['val_loss', 'test_acc', 'test_std', 'time']
    result = pd.DataFrame(np.arange(len(outputs)*len(dataset_name), dtype=np.float32).reshape(
             (len(dataset_name), len(outputs))), index=dataset_name, columns=outputs)

    for dataset in dataset_name:
        loss, acc, std, time, logits, data = cv_run_MLP(dataset, args.gpu_no)
        result.loc[dataset]['loss_mean'] = loss
        result.loc[dataset]['acc_mean'] = acc
        result.loc[dataset]['std_mean'] = std
        result.loc[dataset]['time_mean'] = time
        with open('/home/disk1/xujingyu/桌面/gcn_data/cancer1_MLP.pkl', 'wb') as f:
            pkl.dump([logits, data, acc], f)
        
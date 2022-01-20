#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:12:38 2021

@author: xujingyu
"""
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp

def process_cv_data(model_name):
    """cancatenate the test data across the different cross validation runs.
    Parameters:
    --------
    model_name:    the dataset used and the model used. CPDB_MLP for example
    return:        the cancatened t_true and y_pred
    """
    input_dir = '/home/disk1/xujingyu/桌面/gcn_data'
    input_path = os.path.join(input_dir, '{}.pkl'.format(model_name))
    with open(input_path, 'rb') as f:
        logits, data, test_acc = pkl.load(f)

    y_true,y_pred = [], []
    for i in range(len(logits)):
        pred = logits[i]
        pred = pred.cpu().numpy()
        pred = np.exp(pred)
        pred = pred[:, 1]
        
        test_mask = data[i].test_mask # test mask
        y_true.append(labels[test_mask])
        y_pred.append(pred[test_mask])

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true,y_pred

def process_NRFD_data(dataset):
    input_dir = '/home/disk1/xujingyu/桌面/gcn_data'
    input_path = os.path.join(input_dir, '{}_NRFD.pkl'.format(dataset))
    with open(input_path, 'rb') as f:
        y_pred, y_test = pkl.load(f)

    return y_test,y_pred

def process_pagerank_data(dataset):
    out_path='/home/disk1/xujingyu/桌面/gcn_data/{}_pagerank.txt'.format(dataset)
    result = pd.read_csv(out_path, sep='\t')
    result['Gene_ID'] = result['Gene_ID'].astype('category')
    result['Gene_ID'].cat.reorder_categories(gene_names, inplace=True)
    result.sort_values('Gene_ID', inplace=True)
    logits = result['Score'].values
    
    mask_indices = np.argwhere(~np.isnan(labels))
    y_pred = logits[mask_indices] 
    y_true = labels[mask_indices]
    
    return y_true, y_pred


def process_deepwalk_data(dataset):
    embedding_file='/home/disk1/xujingyu/桌面/gcn_data/embedding_{}'.format(dataset)
    deepwalk_embeddings = pd.read_csv(embedding_file, sep='\t', header=None)
    deepwalk_embeddings.drop(axis=0, index=0, inplace=True)
    deepwalk_embeddings = deepwalk_embeddings.iloc[:, 0].str.split(
        ' ', expand=True)
    deepwalk_embeddings = pd.DataFrame(deepwalk_embeddings, dtype=np.float)

    deepwalk_embeddings.iloc[:, 0] = deepwalk_embeddings.iloc[:, 0].astype(int)
    deepwalk_embeddings.set_index(deepwalk_embeddings.iloc[:, 0], inplace=True)
    deepwalk_embeddings.drop(axis=1, columns=0, inplace=True)
    nodes = pd.DataFrame(gene_names, columns=['Name'])

    X_dw = deepwalk_embeddings.reindex(nodes.index)
    #clf = LogisticRegression(class_weight='balanced')
    #clf = RandomForestClassifier()
    clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
    
    y_pred, y_test = [], []
    for i in range(len(mask_train)):
        X_train_dw = X_dw[mask_train[i].cpu().numpy()]
        X_test_dw = X_dw[mask_test[i].cpu().numpy()]
        y_train = labels[mask_train[i].cpu().numpy()]
        
        clf.fit(X_train_dw, y_train.reshape(-1))
        pred = clf.predict_proba(X_test_dw)
        pred = pred[:, 1]
        y_pred.append(pred)
        y_test.append(labels[mask_test[i]])
        
    y_pred = np.concatenate(y_pred)
    y_test = np.concatenate(y_test)

    return y_test, y_pred


def process_hotnet_data(gene_names):
    input_file = '/home/disk1/xujingyu/桌面/gcn_data/mutation_frequency.txt'
    gene_score = pd.read_table(input_file, header=None)
    gene_score.columns = ['genes', 'score']
    pred = gene_score.iloc[:, 1]
    
    mask_indices = np.argwhere(~np.isnan(labels))
    y_pred = pred[mask_indices]
    y_true = labels[mask_indices]

    return y_true,y_pred

def process_MutSigCv_data(gene_names):
    input_path = '/home/disk1/xujingyu/桌面/gcn_data/mut_output.txt'
    result = pd.read_table(input_path)
    gene_score = result[['gene', 'p']]
    pred = gene_score.iloc[:, 1]
    #pred = pred.fillna(0)
    pred = np.array(pred, dtype=np.float)
    mask_indices = np.argwhere(~np.isnan(labels))
    y_pred = pred[mask_indices]
    y_true = labels[mask_indices]

    return y_true,y_pred

def load_npz_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        edge_index = loader['adj_indices'].copy()
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')
        
        gene_names = loader.get('node_names')

        graph = {
            'A': A,
            'X': X,
            'z': z,
            'n':gene_names
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph

def get_need_paremeter(dataset):
    with open('/home/disk1/xujingyu/桌面/gcn_data/{}_GCN.pkl'.format(dataset), 'rb') as f:
        logits,data,acc = pkl.load(f)
    
    mask_train, mask_test=[], []
    for i in range(len(logits)):
        mask_train.append(data[i].train_mask) #train_mask
        mask_test.append(data[i].test_mask) # test mask
    dataset_path = os.path.join('/home/disk1/xujingyu/DGMP/DGMP/code/data/{}/raw/'.format(dataset),
                                '{}.npz'.format(dataset))
    g = load_npz_dataset(dataset_path)
    labels, gene_names = g['z'],g['n']
    return mask_train, mask_test, labels, gene_names 
 
#mask_train, mask_test, labels, gene_names = get_need_paremeter(dataset='cancer1')
mask_train, mask_test, labels, gene_names = get_need_paremeter(dataset='kegg')

y_true_id, y_pred_id = process_cv_data(model_name='RegNetwork_DGMP')
y_true_nrfd, y_pred_nrfd = process_NRFD_data(dataset='RegNetwork')
y_true_gcn, y_pred_gcn = process_cv_data(model_name='RegNetwork_GCN')
y_true_pr, y_pred_pr = process_pagerank_data(dataset='RegNetwork')
y_true_dw,y_pred_dw = process_deepwalk_data(dataset='RegNetwork')
y_true_hn,y_pred_hn = process_hotnet_data(gene_names)
y_true_ms,y_pred_ms = process_MutSigCv_data(gene_names)

plt.figure(1)
#plt.title('Precision/Recall Curve')  # give plot a title
plt.xlabel('Recall')  # make axis labels
plt.ylabel('Precision')
#plt.xlim((-0.1,1))    
#plt.ylim((0,1))

# DGMP_pr
precision, recall, thresholds = precision_recall_curve(y_true_id, y_pred_id)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color='red',label='DGMP(AUPR={})'.format(round(pr_auc, 2)))

# gcn_pr
precision, recall, thresholds = precision_recall_curve(y_true_gcn, y_pred_gcn)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color='blue',label='EMOGI(AUPR={})'.format(round(pr_auc, 2)))

# nrfd_pr
precision, recall, thresholds = precision_recall_curve(y_true_nrfd, y_pred_nrfd)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color='darkorange',label='NRFD(AUPR={})'.format(round(pr_auc, 2)))

# pagerank_pr
precision, recall, thresholds = precision_recall_curve(y_true_pr, y_pred_pr)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color='green', label='PageRank(AUPR={})'.format(round(pr_auc, 2)))

#deepwalk_pr
precision, recall, thresholds = precision_recall_curve(y_true_dw,y_pred_dw)
pr_auc = auc(recall,precision)
plt.plot(recall, precision, color='cyan',label='DeepWalk+SVM(AUPR={})'.format(round(pr_auc,2)))

#hotnet_pr
precision, recall, thresholds = precision_recall_curve(y_true_hn,y_pred_hn)
pr_auc = auc(recall,precision)
plt.plot(recall, precision, color='yellow',label='HotNet2(AUPR={})'.format(round(pr_auc,2)))

#MutSigCv_pr
precision, recall, thresholds = precision_recall_curve(y_true_ms,y_pred_ms)
pr_auc = auc(recall,precision)
plt.plot(recall, precision, color='grey',label='MutSigCv(AUPR={})'.format(round(pr_auc,2)))

plt.tick_params(axis='both')
plt.legend(bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
plt.show()

plt.figure(2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('roc curve')

# DGMP_roc
fpr, tpr, threshold = roc_curve(y_true_id, y_pred_id)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr,color='red', label='DGMP(AUC={})'.format(round(roc_auc, 2)))

# gcn_roc
fpr, tpr, threshold = roc_curve(y_true_gcn, y_pred_gcn)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label='EMOGI(AUC={})'.format(round(roc_auc, 2)))

# nrfd_roc
fpr, tpr, threshold = roc_curve(y_true_nrfd, y_pred_nrfd)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr,color='darkorange',label='NRFD(AUC={})'.format(round(roc_auc, 2)))

# pagerank_roc
fpr, tpr, threshold = roc_curve(y_true_pr, y_pred_pr)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='green', label='PageRank(AUC={})'.format(round(roc_auc, 2)))

#deepwalk_roc
fpr,tpr,threshold = roc_curve(y_true_dw, y_pred_dw)
roc_auc = auc(fpr,tpr)
plt.plot(fpr, tpr,color='cyan',label= 'DeepWalk+SVM(AUC={})'.format(round(roc_auc,2)))

#hotnet2_roc
fpr,tpr,threshold = roc_curve(y_true_hn, y_pred_hn)
roc_auc = auc(fpr,tpr)
plt.plot(fpr, tpr,color='yellow',label= 'HotNet2(AUC={})'.format(round(roc_auc,2)))

#MutSigCv
fpr,tpr,threshold = roc_curve(y_true_ms, y_pred_ms)
roc_auc = auc(fpr,tpr)
plt.plot(fpr, tpr, color='grey',label= 'MutSigCv(AUC={})'.format(round(roc_auc,2)))

#plt.plot([0,1],[0,1],color='navy', linestyle='--',label = 'random')
plt.tick_params(axis='both')
#plt.legend(loc="lower right")
plt.legend(bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
plt.show()

#fig.savefig(os.path.join(model_dir, 'mean_PR_curve.svg'))
#fig.savefig(os.path.join(model_dir, 'mean_PR_curve.png'), dpi=300)
#fig = plt.figure(figsize=(20, 12))
#plt.tick_params(axis='both', labelsize=20)
#plt.xlabel('Recall', fontsize=25)
#plt.ylabel('Precision', fontsize=25)
#plt.legend(prop={'size':20})
#plt.show()

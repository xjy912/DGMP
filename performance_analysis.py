#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 20:57:22 2021

@author: xujingyu
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import operator
import matplotlib_venn
import scipy.sparse as sp
import argparse
from sklearn.metrics import precision_recall_curve, recall_score, precision_score
import pickle as pkl
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scipy
import networkx as nx

parser = argparse.ArgumentParser(description='parameter for performance analysis')
parser.add_argument('--path', help='path to load data',
                    dest='path',
                    default="/home/disk1/xujingyu/DGMP/DGMP/code/data/RegNetwork/raw/RegNetwork.npz"
                    )
parser.add_argument('--model_dir', help='path to save data',
                    default="/home/disk1/xujingyu/图片"
                    )
parser.add_argument('--topn', help='the top n genes that need to be analysis',
                    dest='topn',
                    default=100
                    )
parser.add_argument('--out_path', help='path to save data',
                    dest='out_path',
                    default='/home/disk1/xujingyu/桌面/gcn_data/'
                    )
args = parser.parse_args()

    
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
    
def get_optimal_cutoff(y_true,y_score,method='PR',colname='Mean_Pred'):
    """Compute an optimal cutoff for classification based on PR curve.

    This method computes optimal an optimal cutoff (the point
    closest to the upper right corner in a PR curve).
    Parameters:
    ----------
    Returns:
    The optimal cutoff as float
    """
    if method == 'PR':
        pr, rec, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_score)
        pr = pr[1:]
        rec = rec[1:]
        thresholds = thresholds[1:]
        distances = np.sqrt(np.sum((np.array([1, 1]) - np.array([rec, pr]).T)**2, axis=1))
        idx = np.argmin(distances)
        best_threshold = thresholds[idx]
        return best_threshold
    elif method == 'IS':
        cutoff_vals = np.linspace(0, .999, 1000)
        all_recall = []
        for cutoff in cutoff_vals:
            r = recall_score(y_true=y_true, y_pred=y_score > cutoff)
            all_recall.append(r)

        all_precision = []
        for cutoff in cutoff_vals:
            p = precision_score(y_true=y_true, y_pred=y_score > cutoff)
            all_precision.append(p)
        diffs = np.abs(np.array(all_precision) - np.array(all_recall))
        return cutoff_vals[diffs.argmin()]
    else:
        print ("Unknown method: {}".format(method))
        return 0.5

def compute_overlap(predictions,set1, set2,threshold=False,names=['DGMP','EMOGI','MTDGMP']):
    """Compute the overlap between predictions and other sets.

    This function computes the overlap between the GCN predictions and two other sets and
    plots a Venn diagram of it.

    Parameters:
    ----------
    model_dir:                  The output directory of the GCN training
    fname_out:                  The filename of the resulting plot
                                (will be written to model_dir)
    predictions:                the predictions of the model
    set1:                       A list, pd.Series or set with probable overlap
    set2:                       Another set that might have overlap with the
                                GCN predictions
    threshold:                  A scalar representing the threshold for the GCN
                                predictions. Default is 0.5
    names:                      Names of set1 and set2 as list of strings.
                                This will be used during plotting.
    """
    fig = plt.figure(figsize=(14, 8))
    if threshold:
        v = matplotlib_venn.venn3([set(set1),
                                   set(predictions[predictions.Prob_pos >= threshold].genes),
                                   set(set2)],
            set_labels=[names[0], names[1]])
    else:
        v = matplotlib_venn.venn3([set(predictions),
                                   set(set1),
                                   set(set2)],
            set_labels=[names[0], names[1], names[2]])
    if not v.get_patch_by_id('10') is None:
        v.get_patch_by_id('10').set_color('#3d3e3d')
        v.get_label_by_id('10').set_fontsize(20)
    if not v.get_patch_by_id('11') is None:
        v.get_patch_by_id('11').set_color('#37652d')
        v.get_label_by_id('11').set_fontsize(20)
    v.get_patch_by_id('011').set_color('#4d2600')
    v.get_label_by_id('A').set_fontsize(20)
    v.get_label_by_id('B').set_fontsize(20)
    v.get_label_by_id('C').set_fontsize(20)
    if not v.get_patch_by_id('01') is None:
        v.get_patch_by_id('01').set_color('#ee7600')
        v.get_label_by_id('01').set_fontsize(20)
    if not v.get_patch_by_id('111') is None and not v.get_patch_by_id('101') is None:
        v.get_label_by_id('111').set_fontsize(20)
        v.get_label_by_id('101').set_fontsize(20)
        v.get_patch_by_id('111').set_color('#890707')
        v.get_patch_by_id('101').set_color('#6E80B7')
    if not v.get_patch_by_id('011') is None:
        v.get_label_by_id('011').set_fontsize(20)
    if not v.get_patch_by_id('001') is None:
        v.get_patch_by_id('001').set_color('#031F6F')
        v.get_label_by_id('001').set_fontsize(20)
    plt.show(fig)
    
def process_result_data(model_name,dataset):
    global node_names
    if model_name=='{}_DGMP'.format(dataset) or model_name=='{}_GCN'.format(dataset):
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
        predictions = logits[0].cpu().numpy()
        predictions = predictions[:,1]
        return y_true,y_pred,predictions
    
def plot_topn_graph(model_name,dataset):
    y_true, y_pred,predictions = process_result_data(model_name,dataset)
    predictions = pd.DataFrame(predictions,columns=['Prob_pos'])
    predictions = node_names.join(predictions,how="left")
    predictions.sort_values(by='Prob_pos',axis=0,ascending=False,inplace=True)
    predictions_topn = predictions.head(args.topn)
    #topn_genes = list(predictions_topn['genes'])
    num_known_in_top =  predictions_topn[predictions_topn.genes.isin(known_cancer_genes)].shape[0]
    num_cand_in_top =  predictions_topn[predictions_topn.genes.isin(candidate_cancer_genes)].shape[0]
    num_onco_in_top =  predictions_topn[predictions_topn.genes.isin(oncokb_no_ncg['Hugo Symbol'])].shape[0]
    print ("Top {} predictions contain {} known driver genes".format(args.topn,num_known_in_top))
    print ("Top {} predictions contain {} candidate driver genes".format(args.topn,num_cand_in_top))
    print ("Top {} predictions contain {} OncoKB genes".format(args.topn,num_onco_in_top))
    
    return predictions
    
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
    adj, labels, gene_names, gene_feature = g['A'],g['z'],g['n'], g['X']
    return mask_train, mask_test,adj, labels, gene_names, gene_feature
 
mask_train, mask_test,adj, labels, gene_names, gene_feature = get_need_paremeter(dataset='RegNetwork')

adj = adj.todense()
train_mask = np.concatenate(mask_train)
test_mask = np.concatenate(mask_test)
node_names = pd.DataFrame(gene_names,columns=['genes'])

ncg_known_cancer_genes = pd.read_table('/home/disk1/xujingyu/桌面/gcn_data/NCG6_tsgoncogene.tsv')
ncg_known_cancer_genes = list(ncg_known_cancer_genes['symbol'])

cgc_known_cancer_genes = pd.read_csv('/home/disk1/xujingyu/桌面/gcn_data/cancer_gene_census.csv',header=0)
cgc_known_cancer_genes = list(cgc_known_cancer_genes['Gene Symbol'])

into_known_cancer_genes = pd.read_table('/home/disk1/xujingyu/桌面/gcn_data/IntOGen-DriverGenes.tsv')
into_known_cancer_genes = list(into_known_cancer_genes['Symbol'])

known_cancer_genes = list(set(ncg_known_cancer_genes)|set(cgc_known_cancer_genes)|set(into_known_cancer_genes))
#known_cancer_genes_innet = node_names[node_names.genes.isin(known_cancer_genes)].genes

ncg_candidate_cancer_genes = pd.read_table('/home/disk1/xujingyu/桌面/gcn_data/NCG6_strong_candidates.tsv')
candidate_cancer_genes = list(ncg_candidate_cancer_genes['symbol'])

oncokb_genes = pd.read_table('/home/disk1/xujingyu/桌面/gcn_data/cancerGeneList.tsv', sep='\t')
oncokb_no_ncg = oncokb_genes[~oncokb_genes['Hugo Symbol'].isin(known_cancer_genes)]
oncokb_no_ncg = oncokb_no_ncg[~oncokb_no_ncg['Hugo Symbol'].isin(candidate_cancer_genes)]
#oncokb_innet = node_names[node_names.genes.isin(oncokb_no_ncg['Hugo Symbol'])]

oncogenes = pd.read_table('/home/disk1/xujingyu/桌面/gcn_data/ongene_human.txt')

predictions = plot_topn_graph(model_name='uncancer1_IDGCN',dataset='uncancer1')
#predictions = plot_topn_graph(model_name='cancer1_MTDGCNMP',dataset='cancer1')
label = pd.DataFrame(labels)
label.columns = ['label']
predictions = predictions.join(label)
predictions_topn = predictions.head(args.topn)

#npcg_plot
kcg = predictions_topn[predictions_topn['label']==1]
npcg = predictions_topn[~predictions_topn.genes.isin(kcg.genes)]
#npcg2 = npcg2[~npcg2.genes.isin(candidate_cancer_genes)]

other = predictions[~predictions.genes.isin(known_cancer_genes)]
other = other[~other.genes.isin(candidate_cancer_genes)]
other = other[~other.genes.isin(npcg.genes)]
other = other[~other.genes.isin(oncokb_genes)]
other = other[~other.genes.isin(oncogenes)]


A = pd.DataFrame(adj, index=gene_names, columns=gene_names)
interactions_with_cancer_genes = A[A.index.isin(known_cancer_genes)].sum(
    axis=0).rename('Cancer_Gene_Interactions')
pred_with_interactions = predictions.join(
    interactions_with_cancer_genes, on='genes')

#pred correlation
fig = plt.figure(figsize=(8, 8))
sns.kdeplot(x=pred_with_interactions.Prob_pos.rank(),y=pred_with_interactions.Cancer_Gene_Interactions.rank(), cmap='Reds',
            shade=True)
correlation, pvalue = scipy.stats.pearsonr(pred_with_interactions.Prob_pos.rank(),
                                           pred_with_interactions.Cancer_Gene_Interactions.rank()
                                            )
plt.text(-1500, 10500, 'R={}'.format(round(correlation, 2)), fontsize=16)
plt.text(-1500, 9800, 'P value<{}'.format(pvalue), fontsize=16)

plt.xlabel('Output Probability (Ranked)', fontsize=16)
plt.ylabel('# Interactions with Cancer Driver genes (Ranked)', fontsize=16)
plt.gca().tick_params(axis='both', labelsize=12)
plt.show()

sns.boxplot(data=pred_with_interactions, x='label', y='Cancer_Gene_Interactions', showfliers=False)
plt.gca().tick_params(axis='x', labelsize=15)
plt.gca().set_xticklabels(['Other', 'Known Cancer\nDriver Genes'])
plt.ylabel('# of Interactions\nwith Known Cancer Driver Genes', fontsize=15)

#npcg_plot
kcg = predictions_topn[predictions_topn['label']==1]
npcg = predictions_topn[~predictions_topn.genes.isin(kcg.genes)]
npcg = npcg[~npcg.genes.isin(candidate_cancer_genes)]
nodes_with_degrees = pred_with_interactions.join((A.sum(axis=1) // 2).rename('Degree'), on='genes')
cg_idx = nodes_with_degrees.drop(['genes', 'Degree', 'label', 'Prob_pos','Cancer_Gene_Interactions'], axis=1).any(axis='columns')
nodes_with_degrees['Cancer_Gene'] = cg_idx
nodes_with_degrees['NPCG'] = False
nodes_with_degrees.loc[nodes_with_degrees.genes.isin(npcg.genes), 'NPCG'] = True 
nodes_with_degrees['Gene_Set'] = 'Other'
nodes_with_degrees.loc[nodes_with_degrees.genes.isin(candidate_cancer_genes), 'Gene_Set'] = 'CCGs'
nodes_with_degrees.loc[nodes_with_degrees.genes.isin(known_cancer_genes), 'Gene_Set'] = 'KCGs'
nodes_with_degrees.loc[nodes_with_degrees.genes.isin(npcg.genes), 'Gene_Set'] = 'NPCGs'

nodes_with_degrees['Other'] = False
nodes_with_degrees.loc[nodes_with_degrees.genes.isin(other.genes), 'Other'] = True 
nodes_with_degrees['CCGs'] = False
nodes_with_degrees.loc[nodes_with_degrees.genes.isin(candidate_cancer_genes), 'CCGs'] = True 
nodes_with_degrees['KCGs'] = False
nodes_with_degrees.loc[nodes_with_degrees.genes.isin(known_cancer_genes), 'KCGs'] = True 

nodes_with_degrees['KCG_interaction_fraction'] = nodes_with_degrees.Cancer_Gene_Interactions / (nodes_with_degrees.Degree + 1)


gene_feature = gene_feature.todense()
gene_feature1 = np.mean(gene_feature[:,0:16],axis=1)
gene_feature1 = np.mean(gene_feature1,axis=1)
gene_feature2 = gene_feature[:,16:32]
gene_feature2 = np.mean(gene_feature2,axis=1)
gene_feature3 = gene_feature[:,32:48]
gene_feature3 = np.mean(gene_feature3,axis=1)
gene_feature4 = gene_feature[:,48:64]
gene_feature4 = np.mean(gene_feature4,axis=1)
gene_feature = np.hstack((gene_feature1,gene_feature2,gene_feature3,gene_feature4))
gene_feature = pd.DataFrame(gene_feature)
gene_feature.columns=['mutation','methylation','express','CNVs']
gene_feature = gene_feature.join(node_names)
nodes_with_feature = pd.merge(nodes_with_degrees,gene_feature, on='genes')

sns.boxplot(data=nodes_with_degrees, x='Gene_Set', y='Degree', showfliers=False)
sns.set_style('whitegrid')
sns.despine(left=True,top=True,right=True,bottom=False)
plt.ylim(0,160)
plt.ylabel('Degree', fontsize=15)
plt.xlabel(None)

#_ = plt.setp(plt.gca().get_xticklabels(), fontsize=15)
#plt.gca().tick_params(axis='y', labelsize=18)
#plt.xlabel(None)

statistic, pval = scipy.stats.mannwhitneyu(nodes_with_degrees[nodes_with_degrees.NPCG].Degree,
                                         nodes_with_degrees[nodes_with_degrees.Other].Degree)

# =============================================================================
# statistic, pval = scipy.stats.ttest_ind(nodes_with_degrees[nodes_with_degrees.NPCG].Degree,
#                                         nodes_with_degrees[nodes_with_degrees.Other].Degree)
# =============================================================================

sns.boxplot(data=nodes_with_degrees, x='Gene_Set', y='KCG_interaction_fraction', showfliers=False)
sns.set_style('whitegrid')
sns.despine(left=True,top=True,right=True,bottom=False)
plt.ylim(0,4)
plt.ylabel('# of Interactions\nwith Known Cancer Driver Genes', fontsize=15)
_ = plt.setp(plt.gca().get_xticklabels(), fontsize=15)
plt.gca().tick_params(axis='y', labelsize=18)
plt.xlabel(None)

statistic, pval = scipy.stats.mannwhitneyu(nodes_with_degrees[nodes_with_degrees.NPCG].KCG_interaction_fraction,
                                         nodes_with_degrees[nodes_with_degrees.Other].KCG_interaction_fraction)

# =============================================================================
# statistic, pval = scipy.stats.ttest_ind(nodes_with_degrees[nodes_with_degrees.NPCG].KCG_interaction_fraction,
#                                         nodes_with_degrees[~nodes_with_degrees.NPCG].KCG_interaction_fraction)
# =============================================================================

sns.boxplot(data=nodes_with_feature, x='Gene_Set', y='mutation', showfliers=False)
sns.set_style('whitegrid')
sns.despine(left=True,top=True,right=True,bottom=False)
plt.ylim(0,0.1)
plt.ylabel('SNVs', fontsize=15)
_ = plt.setp(plt.gca().get_xticklabels(), fontsize=15)
plt.gca().tick_params(axis='y', labelsize=18)
plt.xlabel(None)

statistic, pval = scipy.stats.mannwhitneyu(nodes_with_feature[nodes_with_degrees.NPCG].mutation,
                                         nodes_with_feature[nodes_with_degrees.Other].mutation)

# =============================================================================
# statistic, pval = scipy.stats.ttest_ind(nodes_with_feature[nodes_with_feature.NPCG].mutation,
#                                         nodes_with_feature[nodes_with_feature.Other].mutation)
# =============================================================================

sns.boxplot(data=nodes_with_feature, x='Gene_Set', y='methylation', showfliers=False)
plt.ylabel('DNA methylation', fontsize=15)
sns.set_style('whitegrid')
sns.despine(left=True,top=True,right=True,bottom=False)
plt.ylim(0,0.5)
_ = plt.setp(plt.gca().get_xticklabels(), fontsize=15)
plt.gca().tick_params(axis='y', labelsize=18)
plt.xlabel(None)


statistic, pval = scipy.stats.mannwhitneyu(nodes_with_feature[nodes_with_feature.NPCG].methylation,
                                        nodes_with_feature[nodes_with_feature.Other].methylation)

# =============================================================================
# statistic, pval = scipy.stats.ttest_ind(nodes_with_feature[nodes_with_feature.NPCG].methylation,
#                                         nodes_with_feature[nodes_with_feature.Other].methylation)
# =============================================================================

sns.boxplot(data=nodes_with_feature, x='Gene_Set', y='express', showfliers=False)
plt.ylabel('gene differential expression', fontsize=15)
sns.set_style('whitegrid')
sns.despine(left=True,top=True,right=True,bottom=False)
plt.ylim(0,0.4)
_ = plt.setp(plt.gca().get_xticklabels(), fontsize=15)
plt.gca().tick_params(axis='y', labelsize=18)
plt.xlabel(None)

statistic, pval = scipy.stats.mannwhitneyu(nodes_with_feature[nodes_with_feature.NPCG].express,
                                        nodes_with_feature[nodes_with_feature.Other].express)

sns.boxplot(data=nodes_with_feature, x='Gene_Set', y='CNVs', showfliers=False)
plt.ylabel('CNAs', fontsize=15)
sns.set_style('whitegrid')
sns.despine(left=True,top=True,right=True,bottom=False)
plt.ylim(0,0.4)
_ = plt.setp(plt.gca().get_xticklabels(), fontsize=15)
plt.gca().tick_params(axis='y', labelsize=18)
plt.xlabel(None)

statistic, pval = scipy.stats.mannwhitneyu(nodes_with_feature[nodes_with_feature.NPCG].CNVs,
                                        nodes_with_feature[nodes_with_feature.Other].CNVs)


#plot
fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data=nodes_with_degrees, x='NPCG', y='KCG_interaction_fraction', showfliers=False)
plt.gca().set_xticklabels(['Other', 'NPCGs'], fontsize=20)
plt.xlabel(None)
plt.gca().tick_params(axis='y', labelsize=18)
plt.ylabel('Degree', fontsize=20)

plt.subplot(1, 2, 2)
sns.boxplot(data=nodes_with_degrees, x='NPCG', y='KCG_interaction_fraction', showfliers=False)
plt.ylabel('% of Interactions that\are with KCGs', fontsize=20)
plt.gca().set_xticklabels(['Other', 'NPCGs'], fontsize=20)
plt.gca().tick_params(axis='y', labelsize=18)
plt.xlabel(None)
plt.tight_layout()

#plot
fig = plt.figure(figsize=(10, 7))
kwargs = dict(cumulative=True, linewidth=6, alpha=0.8)
nodes_with_degrees.loc[nodes_with_degrees.KCG_interaction_fraction > 1, 'KCG_interaction_fraction'] = 1
sns.distplot(nodes_with_degrees[nodes_with_degrees.NPCG].KCG_interaction_fraction,
             label='NPCGs', kde_kws=kwargs, hist=False)#, hist_kws=kwargs, )
sns.distplot(nodes_with_degrees[~nodes_with_degrees.NPCG].KCG_interaction_fraction,
              label='Other', kde_kws=kwargs, hist=False) #hist_kws=kwargs, 
plt.xlabel('Fraction of interaction with KCGs',fontsize=18)
plt.ylabel('eCDF percentage',fontsize=18)
plt.legend(loc='upper left')
plt.text(-0.5, 0.85, 'R={}'.format(round(statistic, 2)), fontsize=16)
plt.text(-0.5, 0.8, 'P value<{}'.format( pval ), fontsize=16)
plt.show()



essential_genes = pd.read_csv('/home/disk1/xujingyu/桌面/gcn_data/Achilles_gene_effect.csv').T
essential_genes.columns = essential_genes.iloc[0]
essential_genes.drop('DepMap_ID', inplace=True)
essential_genes['Name'] = [i.split('(')[0].strip() for i in essential_genes.index]
essential_genes.set_index('Name', inplace=True) 
# compute number of affected cell lines for our newly predicted cancer genes
def get_number_of_affected_cellines(gene_name, achilles_data):
    # we only consider negative effects of knockouts because positives are often false
    affected_celllines = (achilles_data[achilles_data.index == gene_name] < -0.5).sum().sum()
    return affected_celllines
 
target_scores = []
for potential_target in npcg.genes:
    num_affected_celllines = get_number_of_affected_cellines(potential_target,  essential_genes)
    target_scores.append((potential_target, num_affected_celllines))
     
#random expectation
average = (essential_genes < -0.5).sum(axis=1).mean()
#median = (essential_genes < -0.5).sum(axis=1).median()
#print (average, median)
#plot
to_plot = target_scores[:30]
fig = plt.figure(figsize=(6, 6))
plt.gca().axvline(average, color='black', lw=5, alpha=.6, ls='--')
sns.barplot(y=[i[0] for i in to_plot], x=[i[1] for i in to_plot], orient='h', color='darkorange')
plt.xlabel('# of affected tumor cell lines', fontsize=16)
plt.gca().tick_params(axis='both', labelsize=12)
fig.tight_layout()

# plot
to_plot = [i for i in target_scores if i[1] > average][:20]
fig = plt.figure(figsize=(6, 6))
plt.gca().axvline(average, color='black', lw=5, alpha=.6, ls='--')
#plt.gca().axvline(median, color='black', lw=5, alpha=.6, ls=':')
sns.barplot(y=[i[0] for i in to_plot], x=[i[1] for i in to_plot], orient='h', color='darkorange')
plt.xlabel('# of affected tumor cell lines', fontsize=16)
plt.gca().tick_params(axis='both', labelsize=16)
fig.tight_layout()

# predictions_topn  = predictions_topn.genes
# outputpath = os.path.join(args.out_path,'THCA_IDGCN_topn.txt')
# predictions_topn.to_csv(outputpath,sep=' ',index=False,header=True)
# =============================================================================

#npcg_plot
#npcg = predictions_topn[~predictions_topn.isin(known_cancer_genes)]
#npcg1 = list(npcg['genes'])

#compute_overlap(npcg3, npcg2,npcg1)


#new_npcg = list(set(npcg2) - set(npcg1) - set(npcg))
#new_npcg = pd.DataFrame(new_npcg)
#common_npcg = list(set(npcg2) & set(npcg1) & set(npcg))
#common_candidate = list(set(common_npcg) & set(candidate_cancer_genes))

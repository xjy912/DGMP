# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:27:20 2021

@author: xjy
"""
import scipy.sparse as sp
import pandas as pd
import numpy as np
import networkx as nx
import random
import os

A = pd.read_table('/home/disk1/xujingyu/DGMP/network/human.source',header = None)
A = A.iloc[0:151215,[0,2]]
A = A.values.tolist()
G = nx.from_edgelist(A)

A1 = pd.read_table('/home/disk1/xujingyu/DGMP/network/RegulatoryDirections/new_kegg.human.reg.direction.txt',sep=' ')
A1 = A1.iloc[:,[0,2]]
A1 = A1.values.tolist()
G1 = nx.from_edgelist(A1)

G.add_nodes_from(G1.nodes(data=True))
G.add_edges_from(G1.edges(data=True))

dic = nx.convert.to_dict_of_dicts(G)
keys = pd.DataFrame(list(dic.keys()))
keys.columns=["genes"]
genes_in_net = keys

#邻接矩阵处理
adj = nx.adjacency_matrix(G)
#G = nx.Graph(adj)
A = sp.csr_matrix(adj)
adj_data = A.data
adj_indices = A.indices
adj_indptr = A.indptr
adj_shape = A.shape

#特征矩阵处理
X = pd.read_table('/home/DGMP/xujingyu/DGMP/multiomics_features.tsv')
#X.columns.values.tolist()                                   #获取列名列表,方便准确更改名称
def get_symbols_from_entrez(list_of_ensembl_ids):
    """Get the hugo gene symbols from a list of Ensembl IDs using mygene.

    This function retrieves hugo gene symbols from Ensembl IDs using
    the mygene python API. It requires a stable internet connection to
    retrieve the annotations.
    @see get_ensembl_from_symbol

    Parameters:
    ----------
    list_of_ensembl_ids:        A list of strings containing the
                                Ensembl IDs
    
    Returns:
    A dataframe containing the mapping between symbols and ensembl IDs.
    If no symbol could be found for an ID, NA is returned in that row.
    The index of the dataframe are the ensembl IDs and the symbols are
    in the other column.
    """
    # get Ensembl IDs for gene names
    mg = mygene.MyGeneInfo()
    res = mg.querymany(list_of_ensembl_ids,
                       scopes='entrezgene',
                       fields='symbol',
                       species='human', returnall=True
                      )

    def get_symbol_and_ensembl(d):
        if 'symbol' in d:
            return [d['query'], d['symbol']]
        else:
            return [d['query'], d['query']]

    node_names = [get_symbol_and_ensembl(d) for d in res['out']]
    # now, retrieve the names and IDs from a dictionary and put in DF
    node_names = pd.DataFrame(node_names, columns=['genes', 'Symbol']).set_index('genes')
    node_names.dropna(axis=0, inplace=True)
    return node_names

X = X.rename(columns = {'Unnamed: 0': 'genes'})
features_in_net = pd.merge(genes_in_net, X, how='left',on='genes')
features_in_net = features_in_net.fillna(0)                   #nan填0，防止错误
features_in_net = features_in_net.iloc[:,1:65].values
X1 = sp.csr_matrix(features_in_net)
attr_data = X1.data
attr_indices = X1.indices
attr_indptr = X1.indptr
attr_shape = X1.shape

#class name
class_names = np.array(['natural_gene','driver_gene'])
node_names = genes_in_net

#标签处理
known_cancer_genes = pd.read_table('/home/disk1/xujingyu/桌面/gcn_data/NCG6_tsgoncogene.tsv')
known_cancer_genes = list(known_cancer_genes['symbol'])

known_cancer_genes_innet = node_names[node_names.genes.isin(known_cancer_genes)].genes

not_positives = node_names[~node_names.genes.isin(known_cancer_genes_innet)]

#/home/disk1/xujingyu/GCN_data/2021.2.22.spydata
# get rid of OMIM genes associated with cancer
omim_cancer_genes = pd.read_excel(io='/home/disk1/xujingyu/桌面/gcn_data/cancer_gene_omim.xlsx',sheet_index=0, header=None)
omim_cancer_genes.columns = ['genes']
not_omim_not_pos = not_positives[~not_positives.genes.isin(omim_cancer_genes.genes)]

# get rid of all the OMIM disease genes
omim_genes = pd.read_csv('C:/Users/xjy/Desktop/GCN_data/GCN_data/OMIM/OMIM/genemap2.txt',
                         sep='\t', comment='#', header=None)
omim_genes.columns = ['Chromosome', 'Genomic Position Start', 'Genomic Position End', 'Cyto Location',
                        'Computed Cyto Location', 'Mim Number', 'Gene Symbol', 'Gene Name',
                        'Approved Symbol', 'Entrez Gene ID', 'Ensembl Gene ID', 'Comments',
                        'Phenotypes', 'Mouse Gene Symbol/ID']
omim_gene_names = []
for idx, row in omim_genes.iterrows():
    gene_names = row['Gene Symbol'].strip().split(',')
    omim_gene_names += gene_names
omim_gene_names = list(set(omim_gene_names))
not_omim_not_pos = not_omim_not_pos[~not_omim_not_pos.genes.isin(omim_gene_names)]

# remove COSMIC highly mutated genes
cosmic_prcoding_mutations = pd.read_csv('C:/Users/xjy/Desktop/GCN_data/GCN_data/CosmicMutantExportCensus.tsv.gz',
                                        compression='gzip', sep='\t',encoding='unicode_escape')
not_omim_cosmic_pos = not_omim_not_pos[~not_omim_not_pos.genes.isin(cosmic_prcoding_mutations['Gene name'])]

# remove genes that belong to KEGG pathways in cancer
kegg_cancer_pathway_genes = pd.read_csv('C:/Users/xjy/Desktop/GCN_data/GCN_data/KEGG_cancer_in_pathways.csv',header=None, names=['genes'])
not_pos_omim_cosmic_kegg = not_omim_cosmic_pos[~not_omim_cosmic_pos.genes.isin(kegg_cancer_pathway_genes.genes)]
kegg_cancer_neighbors_genes = pd.read_csv('C:/Users/xjy/Desktop/GCN_data/GCN_data/KEGG_cancer_neighbors_gene.csv',header=None,names=['genes'])
not_pos_omim_cosmic_kegg = not_pos_omim_cosmic_kegg[~not_pos_omim_cosmic_kegg.genes.isin(kegg_cancer_neighbors_genes.genes)]

#get rid of genes that are not candidate cancer genes
ncg_candidate_cancer_genes = pd.read_table('C:/Users/xjy/Desktop/GCN_data/GCN_data/NCG6_strong_candidates.tsv')
ncg_candidate_cancer_genes = list(ncg_candidate_cancer_genes['symbol'])
negatives = not_pos_omim_cosmic_kegg[~not_pos_omim_cosmic_kegg.genes.isin(ncg_candidate_cancer_genes)]
oncokb_genes = pd.read_table('/home/disk1/xujingyu/桌面/gcn_data/cancerGeneList.tsv', sep='\t')
negatives = negatives[~negatives.isin(oncokb_genes['Hugo Symbol'])]

oncogenes = pd.read_table('/home/disk1/xujingyu/桌面/gcn_data/ongene_human.txt')
negatives = negatives[~negatives.isin(oncogenes['OncogeneName'])]

# =============================================================================
# # remove very low degree genes to lower the bias
# adj = nx.from_numpy_matrix(G)
# di_network = nx.to_pandas_adjacency(adj)
# degrees_with_labels = pd.DataFrame(di_network.sum(),columns=['Degree'])
# degrees_with_labels.index=node_names.genes
# neg_w_degrees = degrees_with_labels[degrees_with_labels.index.isin(negatives.genes)]
# negatives_degnorm = negatives[negatives.genes.isin(neg_w_degrees[neg_w_degrees.Degree >=1].index)]
# =============================================================================

negatives = random.sample(list(negatives.genes), len(known_cancer_genes_innet))

#贴标签
#genes_in_net.loc[583,'genes']='CENPC'
#genes_in_net.label.value_counts(0)
genes_in_net.loc[genes_in_net['genes'].isin(known_cancer_genes_innet),'label'] = 1
genes_in_net.loc[genes_in_net['genes'].isin(negatives),'label'] = 0
labels = genes_in_net.iloc[:,1].values
node_names = node_names.iloc[:,0]

np.savez('RegNetwork.npz',adj_data=adj_data,adj_indices=adj_indices ,adj_indptr=adj_indptr,adj_shape=adj_shape,
                           attr_data=attr_data,attr_indices=attr_indices,attr_indptr=attr_indptr,attr_shape=attr_shape,
                           class_names=class_names,node_names=node_names,labels = labels)
np.load('C:/Users/xujingyu/Desktop/RegNetwrok.npz')

#nan_indices = np.argwhere(np.isnan(labels))
#a = labels.tolist()
#a.count(0)
#sys.path 察看包的路径
#edge_list = nx.to_pandas_edgelist(G)
#outputpath = os.path.join('/home/disk1/xujingyu/DiGCN-main/DiGCN-main/code','THCA_IDGCN_topn.txt')
#edge_list.to_csv(outputpath,sep=' ',index=False,header=True)

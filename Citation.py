import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os

import torch
import sys
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected

from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from torch_scatter import scatter_add
import scipy
from torch_geometric.data import InMemoryDataset, Dataset
from get_adj import get_undirected_adj,get_in_directed_adj,get_out_directed_adj
import sklearn.model_selection

class Citation(InMemoryDataset):
    r"""
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"cora_ml"`,
            :obj:`"citeseer"`, :obj:`"pubmed"`), :obj:`"amazon_computer", :obj:`"amazon_photo", :obj:`"cora_full"`) .
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, name, alpha, cv_run=None,adj_type=None, transform=None, pre_transform=None):
        self.name = name
        self.alpha = alpha
        self.cv_run = cv_run
        self.adj_type = adj_type
        super(Citation, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        return 'data.pt'

    # def download(self):
    #     return

    def process(self):
        data = citation_datasets(self.raw_dir, self.name, self.alpha, self.cv_run, self.adj_type)
        # data = read_planetoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

def citation_datasets(path="/home/disk1/xujingyu/DGMP/DGMP/code/data/RegNetwrok/raw/", dataset='RegNetwrok', alpha=0.1,cv_run=None,adj_type=None):
    # path = os.path.join(save_path, dataset)
    os.makedirs(path, exist_ok=True)
    dataset_path = os.path.join(path, '{}.npz'.format(dataset))
    g = load_npz_dataset(dataset_path)
    adj, features, labels = g['A'], g['X'], g['z']
    
    # Set new random splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * the rest for testing
    
    #single_run
    #mask = train_test_split(labels, seed=1020, train_examples_per_class=240, val_size=500, test_size=None)
    #cancer1
    #mask = train_test_split(labels, seed=1020, train_examples_per_class=300,
    #                       val_examples_per_class=150, test_examples_per_class=225)
    #kegg 
    #mask = train_test_split(labels, seed=1020, train_examples_per_class=300,val_examples_per_class=50, test_examples_per_class=50)
    #CPDB
    #mask = train_test_split(labels, seed=1020, train_examples_per_class=350,val_examples_per_class=50, test_examples_per_class=100)
    
    mask_indices = np.argwhere(~np.isnan(labels))
    mask = np.zeros_like(labels)
    mask[mask_indices] = 1
    k_sets = cross_validation_sets(y=labels, mask=mask, folds=5)
    train_mask, test_mask = k_sets[cv_run]
    mask = {}
    mask['train'] = train_mask
    mask['val'] = test_mask
    mask['test'] = test_mask
    
    mask['train'] = torch.from_numpy(mask['train']).bool()
    mask['val'] = torch.from_numpy(mask['val']).bool()
    mask['test'] = torch.from_numpy(mask['test']).bool()
    
    
    coo = adj.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    features = torch.from_numpy(features.todense()).float()
    
    #features = features[:,16:64]    # remove gene mutation
    #features = np.delete(features, np.s_[16:32], axis=1)  # remove methylation information
    #features = np.delete(features, np.s_[32:48], axis=1)   # remove gene diff_express
    #features = features[:,0:48]   # remove CNVs
    
    x_eye = torch.from_numpy(np.eye(len(features))).float()
    labels = torch.from_numpy(labels).long()
    
    if adj_type == 'or':
        print("Processing in,out and undirect adj matrix")
        indices1 = to_undirected(indices)
        edge_index, edge_weight = get_undirected_adj(indices1, features.shape[0], features.dtype)
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight,y=labels)
        edge_index, edge_weight = get_in_directed_adj(indices, features.shape[0],features.dtype)
        data.edge_index1 = edge_index
        data.edge_weight1 = edge_weight
        edge_index, edge_weight = get_out_directed_adj(indices, features.shape[0],features.dtype)
        data.edge_index2 = edge_index
        data.edge_weight2 = edge_weight
        data.x_eye = x_eye
        
    elif adj_type == 'di':
        print("Processing in,out and undirect adj matrix")
        indices1 = to_undirected(indices)
        edge_index, edge_weight = get_undirected_adj(indices1, features.shape[0],features.dtype)
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight,y=labels)
        edge_index, edge_weight = get_in_directed_adj(indices1, features.shape[0],features.dtype)
        data.edge_index1 = edge_index
        data.edge_weight1 = edge_weight
        edge_index, edge_weight = get_out_directed_adj(indices1, features.shape[0],features.dtype)
        data.edge_index2 = edge_index
        data.edge_weight2 = edge_weight
        data.x_eye = x_eye
        
    elif adj_type == 'un':
        print("Processing to undirected adj")
        indices = to_undirected(indices)
        data = Data(x=features, edge_index=indices, edge_weight=None, y=labels)
    else:
        print("Unsupported adj type.")
        sys.exit()
    
    data.train_mask = mask['train']
    data.val_mask = mask['val']
    data.test_mask = mask['test']

    return data

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

        graph = {
            'A': A,
            'X': X,
            'z': z
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

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = np.nanmax(labels)+1
    num_classes = np.int(num_classes)
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = np.nanmax(labels)+1
    num_classes = np.int(num_classes)
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size, val_size, test_size)

    #print('number of training: {}'.format(len(train_indices)))
    #print('number of validation: {}'.format(len(val_indices)))
    #print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask

def get_y_from_indices(y, mask, indices):
    """Construct vector y and its corresponding mask from label indices.

    This method constructs a vector y and a mask m from some indices given.
    Both returned vectors will be boolean and have shape (n,). y will
    contain 1 at positions where the corresponding node in the GCN is positive
    (disease gene) and mask will contain 1 at positions of the indices.

    Parameters:
    ----------
    y:                  The targets for all nodes
    mask:               The mask for all known nodes
    indices:            The indices selected for the set that is
                        to be constructed.
                        All elements of indices must be in the range (0, y.shape)

    Returns:
    A tuple of two numpy vectors. The first contains the labels and the
    second contains the corresponding mask.
    """
    assert (y.shape[0] == mask.shape[0])
    # construct the mask
    m = np.zeros_like(mask)
    m[indices] = 1
    # construct y
    #y_sub = np.zeros_like(y)
    #y_sub[indices] = y[indices]
    
    return  m

def cross_validation_sets(y, mask, folds):
    """Builds labels and masks for k-fold cross validation.

    Constructs four different sets for k different folds. The four sets
    are all boolean vectors of length n, where n is the number of nodes
    in the GCN and the shape of y and mask.
    The four sets per fold are training labels, testing labels, training
    mask and testing mask, respectively.
    The sets are splitted in a stratified manner, for instance in a 5-fold
    CV, there should be 1/5 of all the positives in the test set.

    Parameters:
    ----------
    y:                  The targets/labels for all nodes. y has to be a
                        binary vector and contain a 1 for positive nodes
                        and 0 for negative nodes.
    mask:               Similar to y, mask has to be a binary vector,
                        containing a 1 for all nodes that are labelled
                        (no matter if positive or negative).
                        y and mask have to have the same first dimension.
    folds:              The number of folds to split the data to, i.e. k

    Returns:
    A list of length `folds` or k. Each of the elements in the list is a tuple
    of length four, containing the training labels, testing labels, training
    mask and testing mask, respectively.
    """
    label_idx = np.where(mask == 1)[0] # get indices of labeled genes
    kf = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=False)
    splits = kf.split(label_idx, y[label_idx])
    k_sets = []
    for train, test in splits:
        # get the indices in the real y and mask realm
        train_idx = label_idx[train]
        test_idx = label_idx[test]
        # construct y and mask for the two sets
        train_mask = get_y_from_indices(y, mask, train_idx)
        test_mask = get_y_from_indices(y, mask, test_idx)
        k_sets.append((train_mask, test_mask))
    return k_sets

if __name__ == "__main__":
    data = citation_datasets(path="/home/disk1/xujingyu/DGMP/DGMP/code/data/RegNetwork/raw/", dataset='RegNetwrok')
    print(data.train_mask.shape)
    # print_dataset_info()
    # get_npz_data(dataset='amazon_photo')
    ### already fixed split dataset!!!
    #if opt.dataset == 'all':
    #    for mode in ['cora', 'cora_ml','citeseer','dblp','pubmed']:
    #        get_npz_data(dataset = mode)
    #else:
    #    get_npz_data(dataset = opt.dataset)
    
#super  调用父类方法
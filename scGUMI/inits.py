import os
import torch
import random
import time
import numpy as np
import numpy
import pandas as pd
import scanpy as sc
import pickle
import scipy.sparse as sp
from torch.backends import cudnn
from preprocess import read_data
#from preprocess_atac import read_data
import pandas as pd
from sklearn.neighbors import kneighbors_graph
#from muon import atac as ac

def load_data(args):
    """Load data"""
    #read data
    data = read_data(args) 
      
    print('Data reading finished!')
    return data 

def renew_adata_graph_prot(adata_3, adata_4):

    sc.pp.pca(adata_3, n_comps=50) #22
    feat_1 = adata_3.obsm['X_pca']
    adata_3.obsm['feat'] = feat_1

    sc.pp.pca(adata_4, n_comps=50) #50
    feat_2 = adata_4.obsm['X_pca']
    adata_4.obsm['feat'] = feat_2   

    def construct_graph_by_feature(express1, express2, k=20, mode= "connectivity", metric="correlation", include_self=False):
        graph_omics1=kneighbors_graph(express1, k, mode=mode, metric=metric, include_self=include_self)
        graph_omics2=kneighbors_graph(express2, k, mode=mode, metric=metric, include_self=include_self)

        return graph_omics1, graph_omics2

    graph_omics1, graph_omics2 = construct_graph_by_feature(feat_1, feat_2)
    adata_3.obsm['graph_feat'], adata_4.obsm['graph_feat'] = graph_omics1, graph_omics2

    data = {'omics1': adata_3, 'omics2': adata_4}

    return data


def renew_adata_graph_atac(adata_3, adata_4):

    sc.pp.pca(adata_3, n_comps=50) #22
    feat_1 = adata_3.obsm['X_pca']
    adata_3.obsm['feat'] = feat_1

    ac.tl.lsi(adata_4)
    adata_4.obsm['X_lsi'] = adata_4.obsm['X_lsi'][:,1:]
    adata_4.varm['LSI'] = adata_4.varm['LSI'][:,1:]
    adata_4.uns['lsi']['stdev'] = adata_4.uns['lsi']['stdev'][1:]
    feat_2 = adata_4.obsm['X_lsi']
    adata_4.obsm['feat'] = feat_2 

    def construct_graph_by_feature(express1, express2, k=20, mode= "connectivity", metric="correlation", include_self=False):
        graph_omics1=kneighbors_graph(express1, k, mode=mode, metric=metric, include_self=include_self)
        graph_omics2=kneighbors_graph(express2, k, mode=mode, metric=metric, include_self=include_self)

        return graph_omics1, graph_omics2

    graph_omics1, graph_omics2 = construct_graph_by_feature(feat_1, feat_2)
    adata_3.obsm['graph_feat'], adata_4.obsm['graph_feat'] = graph_omics1, graph_omics2

    data = {'omics1': adata_3, 'omics2': adata_4}

    return data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)          
            
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)  #将列表转为张量

def fix_seed(seed):
    #seed = 666
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized
    
  


   
        
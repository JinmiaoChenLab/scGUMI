import os
import torch
import argparse
import warnings
import time
from train import Train
from inits import load_data, fix_seed, renew_adata_graph_prot, renew_adata_graph_atac
from utils import UMAP, plot_weight_value, clustering
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import scipy
#from muon import atac as ac
from sklearn.neighbors import kneighbors_graph
import pickle

parser = argparse.ArgumentParser(description='PyTorch implementation of spatial multi-omics data integration')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')  # 0.001
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.') # 1000
parser.add_argument('--weight_decay', type=float, default=0.0000, help='Weight for L2 loss on embedding matrix.')  #5e-4
parser.add_argument('--dataset', type=str, default='Mouse_Embryo_E11', help='Dataset tested.')
parser.add_argument('--random_seed', type=int, default=2022, help='Random seed') # 50
# parser.add_argument('--dim_input', type=int, default=3000, help='Dimension of input features') # 100
parser.add_argument('--dim_input_RNA', type=int, default=3000, help='Dimension of input features')
parser.add_argument('--dim_input_PROT', type=int, default=3000, help='Dimension of input features')
parser.add_argument('--dim_output', type=int, default=32, help='Dimension of output features') # 64
parser.add_argument('--n_neighbors', type=int, default=6, help='Number of sampling neighbors') # 6
parser.add_argument('--n_clusters', type=int, default=58, help='Number of clustering') # 3

parser.add_argument('--n_i_RNA', type=int, default=58, help='Number of clustering') # 3
parser.add_argument('--n_h_RNA', type=int, default=58, help='Number of clustering') # 3
parser.add_argument('--n_o_RNA', type=int, default=58, help='Number of clustering') # 3
parser.add_argument('--n_i_PROT', type=int, default=58, help='Number of clustering') # 3
parser.add_argument('--n_h_PROT', type=int, default=58, help='Number of clustering') # 3
parser.add_argument('--n_o_PROT', type=int, default=58, help='Number of clustering') # 3
parser.add_argument('--data_type', type=str, default='RNA_ADT', help='Type of data') # 3

parser.add_argument('--mask_mod1', action='store_true', help='Type of data') # 3
parser.add_argument('--mask_mod2', action='store_true', help='Type of data') # 3
parser.add_argument('--training_exclude_mask', action='store_true', help='Type of data') # 3

args = parser.parse_args([
                "--learning_rate", "0.0001",
                "--epochs", "600",
                "--weight_decay", "0",
                "--input", "D:/data/",
                "--output", "/data/behmoaras/home/e1139777/scGLUE/Data/",
                "--n_neighbors", "10",
                "--random_seed", "50",

                "--dim_output", "64",

                "--n_i_RNA", "3000",
                "--n_h_RNA", "256",
                "--n_o_RNA", "128",
                # "--n_i_PROT", "25",
                "--n_h_PROT", "256", #only use for ATAC
                "--n_o_PROT", "128",
    
                # "--mask_mod1",
                # "--mask_mod2",
                # "--training_exclude_mask",
            ])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# t = time.time()
fix_seed(args.random_seed)

args.dataset = 'CBMC'
args.data_type = 'RNA_ADT'

data = load_data(args) 
adata_1, adata_2 = data['omics1'], data['omics2']

trainer = Train(args, device, data) 

feat1_after, feat2_after, emb_combined, alpha_omics, emb_1_within, emb_2_within= trainer.train()

adata_1.obsm['feat1_after'] = feat1_after
adata_2.obsm['feat2_after'] = feat2_after

adata_1.obsm['emb_combined'] = emb_combined
adata_2.obsm['emb_combined'] = emb_combined

adata_1.obsm['alpha'] = alpha_omics
 
result = {'omics1': adata_1, 'omics2': adata_2}

with open(args.output + args.dataset + '/result1.pkl', 'wb') as file:
    pickle.dump(result, file) 
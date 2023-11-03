import scipy
import anndata
import sklearn
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional, Union
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 
from sklearn.decomposition import PCA
import muon as mu
from muon import atac as ac

def read_data(args):
    """
    Reading adata
    """
    
    # CBMC
    adata_1 = sc.read_h5ad("./scGUMI/Data/CBMC/RNA_cd4.h5ad") # RNA
    adata_2 = sc.read_h5ad("./scGUMI/Data/CBMC/ADT_cd4.h5ad") # ADT
    
    adata_1.var_names_make_unique()
    adata_2.var_names_make_unique()
       
    ##########################################pre-processing for RNA##################################################
    # preprocess before scale
    # select high variable gene
    sc.pp.highly_variable_genes(adata_1, flavor="seurat_v3", n_top_genes=3000)

    # log_normalize the count
    sc.pp.normalize_total(adata_1, target_sum=1e4)
    sc.pp.log1p(adata_1)
    
    adata_1 =  adata_1[:, adata_1.var['highly_variable']]
    
    # filter out HVG
    # adata_1 =  adata_1[:, adata_1.var['highly_variable']]
    adata_1.obsm['X_original'] = adata_1.X.copy()
    
    
    #########################################pre-processing_for_ADT_or_ATAC##########################################
    # preprocess before scale
    # process ADT or ATAC
    if args.data_type == 'RNA_ADT':
        # pre-processing ADT
        adata_2 = clr_normalize_each_cell(adata_2)
        
        
    else:
        # pre-processing ATAC
        sc.pp.highly_variable_genes(adata_2, flavor="seurat_v3", n_top_genes=5000)

        # ac.pp.tfidf(adata_2, scale_factor=1e4)
        
        adata_2.X = tfidf(adata_2.X).toarray()
        
        # sc.pp.normalize_total(adata_2, target_sum=1e4)
        # sc.pp.log1p(adata_2)
        adata_2 = adata_2[:, adata_2.var['highly_variable']]
        
    adata_2.obsm['X_original'] = adata_2.X.copy()
        
    
    #############################################mask_data_to_imputation#############################################
    if args.mask_mod1 == True:
        
        
        # save original count matrix
        
        mask = np.ones(adata_1.X.shape)
        
        # mask the count matrix to imputation
        processed_count_matrix = adata_1.X.copy()
        
        idxi, idxj = np.nonzero(processed_count_matrix)
        masking_percentage = 0.9
        ix = np.random.choice(len(idxi), int(np.floor(masking_percentage * len(idxi))), replace = False) # masking only on the cells.
        
        mask[idxi[ix], idxj[ix]] = 0
        
        # adata_1.X = adata_1.X * mask
        adata_1.X = adata_1.X.toarray() * mask
        # processed_count_matrix[idxi[ix], idxj[ix]] = 0  # making masks 0
        
        # adata_1.X = processed_count_matrix.copy()
        
        # save idxi, idxj, ix to a file
        # np.savetxt(args.output + args.dataset + '/idxi.txt', idxi, fmt='%d')
        # np.savetxt(args.output + args.dataset +'/idxj.txt', idxj, fmt='%d')
        # np.savetxt(args.output + args.dataset +'/ix.txt', ix, fmt='%d')
        
        adata_1.obsm['X_masked'] = adata_1.X.copy()
        
        
        np.savetxt(args.output + args.dataset +'/mask.txt', mask, fmt='%d')

        
    if args.mask_mod2 == True:
        
        
        # save original count matrix
        
        mask = np.ones(adata_2.X.shape)
        
        # mask the count matrix to imputation
        processed_count_matrix = adata_2.X.copy()
        
        idxi, idxj = np.nonzero(processed_count_matrix)
        masking_percentage = 0
        ix = np.random.choice(len(idxi), int(np.floor(masking_percentage * len(idxi))), replace = False) # masking only on the cells.
        
        mask[idxi[ix], idxj[ix]] = 0
        
        # adata_2.X = adata_2.X.toarray() * mask
        adata_2.X = adata_2.X * mask
    
        
        np.savetxt(args.output + args.dataset +'/mask2.txt', mask, fmt='%d')
        adata_2.obsm['X_masked'] = adata_2.X.copy()
    
    
    
    
    #############################################continue_process####################################################
    # scale the RNA count
    sc.pp.scale(adata_1, max_value=10)
    # sc.pp.normalize_total(adata_1, target_sum=1e4)
    # sc.pp.log1p(adata_1)
    adata_1.obsm['X_masked_scaled'] = adata_1.X.copy()
    
    # save the mean and var for future reconstruct
    
    # different PCA selection based on the protein dimension
    if args.data_type == 'RNA_ADT':
        n_pro = adata_2.n_vars - 1
        
        if adata_2.n_vars < 51:
            n_rna = 50
        else:
            n_rna = n_pro
            
    else:
        n_rna = 50
        # n_pro = 50
        
    pca = PCA(n_components = n_rna)
    pca.fit(adata_1.X)
    X_pca = pca.transform(adata_1.X)
    # pca.fit(adata_1.X.toarray())
    # X_pca = pca.transform(adata_1.X.toarray())
    
    adata_1.obsm["X_pca"] = X_pca.copy()
    adata_1.obsm['feat'] = X_pca.copy()
    
    
    #scale the ADT and ATAC count
    if args.data_type == 'RNA_ADT':
        sc.pp.scale(adata_2)

        pca = PCA(n_components = n_pro)
        pca.fit(adata_2.X)
        X_pca = pca.transform(adata_2.X)
        
        adata_2.obsm["X_pca"] = X_pca.copy()
        adata_2.obsm['feat'] = X_pca.copy()
        
    else:
        
        lsi(adata_2)
        # ac.tl.lsi(adata_2)

        # adata_2.obsm['X_lsi'] = adata_2.obsm['X_lsi'][:,1:]
        # adata_2.varm["LSI"] = adata_2.varm["LSI"][:,1:]
        # adata_2.uns["lsi"]["stdev"] = adata_2.uns["lsi"]["stdev"][1:]
        
        sc.pp.scale(adata_2)
        
        # sc.tl.pca(adata_2)
        adata_2.obsm['feat'] = adata_2.obsm['X_lsi'].copy()
        # adata_2.obsm['feat'] = adata_2.obsm['X_pca'].copy()
    
    
    ############################################## buid graph by feature ###########################################
    graph_omics1, graph_omics2 = construct_graph_by_feature(adata_1.obsm['feat'], adata_2.obsm['feat'])
    adata_1.obsm['graph_feat'], adata_2.obsm['graph_feat'] = graph_omics1, graph_omics2
    
    data = {'omics1': adata_1, 'omics2': adata_2}
    
    # print(n_rna)
    print('omics1:', adata_1)
    print('omics2:', adata_2)
    
    return data



def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else adata.X)
    )
    return adata 

def construct_graph_by_feature(express1, express2, k=30, mode= "connectivity", metric="correlation", include_self=False):
    graph_omics1=kneighbors_graph(express1, k, mode=mode, metric=metric, include_self=include_self)
    graph_omics2=kneighbors_graph(express2, k, mode=mode, metric=metric, include_self=include_self)

    return graph_omics1, graph_omics2

def lsi(adata):
    
#     sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=30000)

#     adata.X = tfidf(adata.X).toarray()
    
#     # sc.pp.highly_variable_genes(adata, min_mean=0.05, max_mean=1.5, min_disp=.5)
#     adata = adata[:, adata.var['highly_variable']]
    
    X = adata.X
    
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, 50)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]
    
    return adata

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf



       
    
          
  
   
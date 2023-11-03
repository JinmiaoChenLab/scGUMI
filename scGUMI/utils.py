import os
import pickle
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['R_HOME'] = 'C:/PROGRA~1/R/R-42~1.3'    
    
def UMAP(adata_s, adata_t, args, size1=10, size2=20, resolution=0.38): #10, 20
    # -------------------   plotting UMAP   ----------------------------
    fig, ax_list = plt.subplots(2, 2, figsize=(7, 5))
    ## UMAP on original feature
    sc.pp.neighbors(adata_s, use_rep='X_pca', n_neighbors=10)
    sc.tl.umap(adata_s, min_dist=0.8)
    #sc.tl.leiden(adata_s, resolution=resolution, key_added='leiden_origi')
    sc.pl.umap(adata_s, color='cluster', ax=ax_list[0, 0], title='mRNA', s=size1, show=False)
    
    sc.pp.neighbors(adata_t, use_rep='X_pca', n_neighbors=10)
    sc.tl.umap(adata_t, min_dist=0.8)
    #sc.tl.leiden(adata_t, resolution=resolution, key_added='leiden_origi')
    sc.pl.umap(adata_t, color='cluster', ax=ax_list[1, 0], title='Protein', s=size1, show=False)
   
    ## UMAP on latent representation
    sc.pp.neighbors(adata_s, use_rep='emb_combined', n_neighbors=10)
    sc.tl.umap(adata_s, min_dist=0.8)
    sc.tl.leiden(adata_s, resolution=resolution, key_added='scGlue')
    sc.pl.umap(adata_s, color='scGlue', ax=ax_list[0, 1], title='scGlue', s=size1, show=False)
    
    sc.pl.umap(adata_s, color='cluster', ax=ax_list[0, 1], title='scGlue+Know', s=size1, show=False)
    
    # save adata_combined
    #save_path = '/home/yahui/anaconda3/work/SpatialGlue_omics/output/' + args.dataset + '/'
    #adata_s.write_h5ad(save_path + 'adata_combined.h5ad')
    
    plt.tight_layout(w_pad=0.3)
    plt.show()
    
    #save_path = '/home/yahui/anaconda3/work/SpatialGlue_omics/output/' + args.dataset + '/'
    #plt.savefig(save_path + args.dataset + '.jpg', bbox_inches='tight', dpi=300)
    
    return adata_s
    
def plot_hist(lost):
    #ax = plt
    #print('Plotting loss')
    #print(lost)
    values = np.array(lost)
    size = values.size
    values = values.flatten()
    #values = np.log10(values.flatten())
    plt.plot(np.arange(0, size), values)
    plt.show()

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, keys='emb_pca', add_keys='label', n_clusters=10):
    #pca = PCA(n_components=20, random_state=42) 
    #embedding = pca.fit_transform(adata.obsm['emb_sp'])
    #adata.obsm['emb_sp_pca'] = embedding
    adata = mclust_R(adata, used_obsm=keys, num_cluster=n_clusters)
    adata.obs[add_keys] = adata.obs['mclust']
    #adata = refine_label(adata, args)
    
    #return adata

def plot_weight_value(alpha, label, figsize):
  import pandas as pd  
  
  df = pd.DataFrame(columns=['weight1','weight2','label'])  
  df['weight1'], df['weight2'] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_combined', 'View', 'Weight value']
#   ax = sns.violinplot(data=df, x='label_combined', y='Weight value', hue="View",
#                 split=True, inner="quart", linewidth=1, show=True)
#   ax.set_title('RNA vs ADT')

  plt.figure(figsize=figsize)
  sns.violinplot(data=df, x='label_combined', y='Weight value', hue="View",
            split=True, inner="quart", linewidth=1, show=True)
  plt.show()
  path = 'D:/Data/' 
  plt.savefig(path + 'alpha.jpg', bbox_inches='tight', dpi=300)    

import torch
from model import Encoder_omics
from inits import get_edge_index, preprocess_adj, preprocess_graph
from torch import nn
# from preprocess import construct_graph
import torch.nn.functional as F
from utils import plot_hist
#import pickle
from tqdm import tqdm
#import scipy.sparse as sp
import numpy as np
#import ot
from scipy.sparse import coo_matrix
# from mnn import create_dictionary_mnn, create_dictionary_cosine

class Train:
    def __init__(self, args, device, data):
        self.args = args
        self.device = device
        self.data = data.copy()
        self.adata_1 = self.data['omics1']
        self.adata_2 = self.data['omics2']
        
        self.n_omics1 = self.adata_1.n_obs
        self.n_omics2 = self.adata_2.n_obs

        # self.features_1 = torch.FloatTensor(self.adata_1.X.toarray().copy()).to(self.device)
        # self.features_2 = torch.FloatTensor(self.adata_2.X.toarray().copy()).to(self.device)
        if args.data_type == 'RNA_ADT':
           self.features_1 = torch.FloatTensor(self.adata_1.X.copy()).to(self.device)
           self.features_2 = torch.FloatTensor(self.adata_2.X.copy()).to(self.device)
        else:
            # self.features_1 = torch.FloatTensor(self.adata_1.X.toarray().copy()).to(self.device)
            # self.features_2 = torch.FloatTensor(self.adata_2.X.toarray().copy()).to(self.device)
           self.features_1 = torch.FloatTensor(self.adata_1.X.copy()).to(self.device)  # 3000
           self.features_2 = torch.FloatTensor(self.adata_2.X.copy()).to(self.device)   # 50
        
        # # feature
        #self.features_1 = torch.FloatTensor(self.adata_1.obsm['feat'].copy()).to(self.device)
        #self.features_2 = torch.FloatTensor(self.adata_2.obsm['feat'].copy()).to(self.device)
        
        # # dimension of input feature
        # self.args.dim_input = self.features_1.shape[1]
        # self.args.dim_output = self.args.dim_output

        
        # ######################################## adj ########################################
        self.graph_1 = torch.FloatTensor(self.adata_1.obsm['graph_feat'].copy().toarray())
        self.graph_2 = torch.FloatTensor(self.adata_2.obsm['graph_feat'].copy().toarray())
        
        self.graph_1 = self.graph_1 + self.graph_1.T
        self.adj1 = np.where(self.graph_1>1, 1, self.graph_1)
        self.graph_2 = self.graph_2 + self.graph_2.T
        self.adj2 = np.where(self.graph_2>1, 1, self.graph_2)
        
        # Sparse version
        self.adj_1 = preprocess_graph(self.adj1).to(self.device) # view 2 (by feature)
        self.adj_2 = preprocess_graph(self.adj2).to(self.device)
        
        #self.args.dim_input_RNA = self.features_1.shape[1]
        #self.args.dim_input_PROT = self.features_2.shape[1]
        
        self.args.n_i_PROT = self.features_2.shape[1]
        
        self.beloss = nn.BCELoss()
        self.margin = 1.0
        
        self.current_loss_2 = None
        
        if self.args.training_exclude_mask == True:
            self.mask = torch.FloatTensor(np.loadtxt('./scGUMI/Data/' + args.dataset + '/mask.txt', dtype=int)).to(self.device)
            self.mask2 = torch.FloatTensor(np.loadtxt('./scGUMI/Data/' + args.dataset + '/mask2.txt', dtype=int)).to(self.device)
        
    def train(self):
        self.model = Encoder_omics(self.args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.learning_rate, 
                                          weight_decay=self.args.weight_decay)
        self.model.train()
        hist = []
        hist_mod1 = []
        hist_mod2 = []
        hist_reg = []
        hist_delta = []
        
        ## PBMC
        #emb_1 = self.pretrain(self.features_1, self.adj_1)
        #emb_2 = self.pretrain(self.features_2, self.adj_2)
        
        ## use raw feature
        #emb_1 = torch.FloatTensor(self.adata_1.obsm['X_pca'].copy())
        #emb_2 = torch.FloatTensor(self.adata_2.obsm['X_pca'].copy())
        
        # hist = []
        self.model.train()
        for epoch in tqdm(range(self.args.epochs)): #60
            self.model.train()
        
            self.emb_1_within, self.emb_2_within, self.emb_combined, _, self.feat1_after, self.feat2_after, self.emb1, self.emb2\
                = self.model(self.features_1, self.features_2, self.adj_1, self.adj_2)
            print(self.feat1_after.shape)
            print(self.feat2_after.shape)
            
            
            # reconstruction loss    
            if self.args.training_exclude_mask == True:
                
                # idxi, idxj = torch.where(self.mask == 1)
                self.loss_1_combined = ((self.features_1 - self.feat1_after) ** 2 * self.mask).mean()
                # self.loss_1_combined = ((self.features_1 - self.feat1_after) ** 2)[idxi,idxj].mean()
                
                self.loss_2_combined = ((self.features_2 - self.feat2_after) ** 2 * self.mask2).mean()
                
            else:
                self.loss_1_combined = F.mse_loss(self.features_1, self.feat1_after)
                self.loss_2_combined = F.mse_loss(self.features_2, self.feat2_after)
            # self.loss_2_combined = F.mse_loss(self.features_2, self.feat2_after)
            
            
            
            # add L1 and L2 regulation
            model_parameters = torch.cat([x.view(-1) for x in self.model.parameters()])
            l1_regularization = torch.norm(model_parameters, 1)
            l2_regularization = torch.norm(model_parameters, 2)
            self.loss_reg = 0.0001 * l1_regularization + 0.0001 * l2_regularization

            
            
            if self.args.data_type == 'RNA_ADT':
                loss = 100 * self.loss_1_combined + 1 * self.loss_2_combined + 0.1*self.loss_reg
            else:
                loss = 1 * self.loss_1_combined + 1 * self.loss_2_combined #+ 0.001*self.loss_reg
                    
                    
                    
            
            print('self.loss_1_combined:', self.loss_1_combined)
            print('self.loss_2_combined:', self.loss_2_combined)
            print('self.loss_reg:', self.loss_reg)
            print('loss:', loss)
            
            
            
            hist.append(loss.data.cpu().numpy())
            hist_reg.append(self.loss_reg.data.cpu().numpy())
            hist_mod1.append(self.loss_1_combined.data.cpu().numpy())
            hist_mod2.append(self.loss_2_combined.data.cpu().numpy())
            
            
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        
        # plot training loss
        plot_hist(hist)
        plot_hist(hist_mod1)
        plot_hist(hist_mod2)
        plot_hist(hist_reg)
        # plot_hist(hist_delta)
        print("Training finished!\n")    
        
        with torch.no_grad():
          self.model.eval()
          _, _, emb_combined, alpha_omics, feat1_after, feat2_after, emb_1_within, emb_2_within \
            = self.model(self.features_1, self.features_2, self.adj_1, self.adj_2)
          
        # feat1_after = F.normalize(feat1_after, p=2, eps=1e-12, dim=1)  
        # feat2_after = F.normalize(feat2_after, p=2, eps=1e-12, dim=1)
        # emb_combined = F.normalize(emb_combined, p=2, eps=1e-12, dim=1)
        
        
        
        return feat1_after.detach().cpu().numpy(), feat2_after.detach().cpu().numpy(), emb_combined.detach().cpu().numpy(), alpha_omics.detach().cpu().numpy(), \
                emb_1_within.detach().cpu().numpy(), emb_2_within.detach().cpu().numpy()
        
    
    
      

    
        
    
    

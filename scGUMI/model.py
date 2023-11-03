import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
from sklearn.neighbors import kneighbors_graph 
from inits import get_edge_index, preprocess_adj, preprocess_graph
from torch.autograd import Function
#from torch_geometric.nn import GCNConv, GATConv

class ReverseLayerF(Function):

    #@staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha

        return x.view_as(x)

    #@staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None  
    
class Encoder_omics(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, args, dropout=0.0, act=F.relu):
        super(Encoder_omics, self).__init__()
        self.args = args
        # self.in_feat = self.args.dim_input
        #self.in_feat_RNA = self.args.dim_input_RNA
        #self.in_feat_PROT = self.args.dim_input_PROT
        self.in_feat_RNA = self.args.n_o_RNA
        self.in_feat_PROT = self.args.n_o_PROT
        self.out_feat = self.args.dim_output
        self.dropout = dropout
        self.act = act
        

        self.MLP1 = MLP(self.args.n_i_RNA, self.args.n_h_RNA, self.args.n_o_RNA)  #128
        
        if self.args.data_type == 'RNA_ADT':
            self.MLP2 = MLP2(self.args.n_i_PROT, self.args.n_h_PROT, self.args.n_o_PROT) #128
        else:
            self.MLP2 = MLP(self.args.n_i_PROT, self.args.n_h_PROT, self.args.n_o_PROT) #128

        self.encoder_omics1 = Encoder(self.args.n_o_RNA, self.out_feat)
        self.encoder_omics2 = Encoder(self.args.n_o_PROT, self.out_feat)

        self.atten_cross = AttentionLayer(self.out_feat, self.out_feat)
        
        if self.args.training_exclude_mask == True:
            self.decoder_omics1 = Decoder_fc(self.out_feat, self.in_feat_RNA)
            self.decoder_omics2 = Decoder_fc(self.out_feat, self.in_feat_PROT)
        else:    
            self.decoder_omics1 = Decoder(self.out_feat, self.in_feat_RNA)
            self.decoder_omics2 = Decoder(self.out_feat, self.in_feat_PROT)
        # self.decoder_omics1 = Decoder(self.out_feat, self.in_feat_RNA)
        # self.decoder_omics2 = Decoder(self.out_feat, self.in_feat_PROT)

        self.MLP3 = MLP(self.args.n_o_RNA, self.args.n_h_RNA, self.args.n_i_RNA)
        
        
        if self.args.data_type == 'RNA_ADT':
            self.MLP4 = MLP2(self.args.n_o_PROT, self.args.n_h_PROT, self.args.n_i_PROT)
        else:
            self.MLP4 = MLP(self.args.n_o_PROT, self.args.n_h_PROT, self.args.n_i_PROT)
            

        
    def forward(self, omic1, omic2, adj_1, adj_2):

        feat1 = self.MLP1(omic1)
        feat2 = self.MLP2(omic2)
        #feat1 = omic1
        #feat2 = omic2

        # encoder
        emb_1_within = self.encoder_omics1(feat1, adj_1)
        emb_2_within = self.encoder_omics2(feat2, adj_2)
    
        # attention integration
        emb_combined, alpha_omics = self.atten_cross(emb_1_within, emb_2_within)
        
        # decoder
        emb_1_decoder = self.decoder_omics1(emb_combined, adj_1)
        emb_2_decoder = self.decoder_omics2(emb_combined, adj_2)
        
        
        
        # reconstruction
        feat1_after = self.MLP3(emb_1_decoder)
        feat2_after = self.MLP4(emb_2_decoder)
        
      
        return emb_1_within, emb_2_within, emb_combined, alpha_omics, feat1_after, feat2_after, emb_1_within, emb_2_within  

class Discriminator(nn.Module):
    """Latent space discriminator"""
    def __init__(self, dim_input, n_hidden=50, n_out=1):
        super(Discriminator, self).__init__()
        self.dim_input = dim_input
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(dim_input, n_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_hidden, 2*n_hidden),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(n_hidden, n_hidden),
            #nn.ReLU(inplace=True),
            #nn.Linear(n_hidden, n_hidden),
            #nn.ReLU(inplace=True),
            #nn.Linear(n_hidden, n_hidden),
            #nn.ReLU(inplace=True),
            nn.Linear(2*n_hidden,n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
    
class MLP(Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(num_i, num_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_h, num_o)
        # self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        # x = self.relu(x)
        x = self.linear2(x)
        # x = self.relu2(x)
        return x
    
class MLP2(Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP2, self).__init__()
        self.linear1 = nn.Linear(num_i, num_o)
        
    def forward(self, x):
        x = self.linear1(x)
        return x

class Encoder(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight1)
        emb = torch.spmm(adj, x)
        
        return emb
    

class Decoder(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight2 = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight2)
        
    def forward(self, feat, adj):
        
        x = torch.mm(feat, self.weight2)
        emb = torch.spmm(adj, x)
        
        return emb
    
    
    
class Decoder_fc(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder_fc, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.linear = nn.Linear(in_feat, out_feat)

        self.weight2 = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight2)
        
    def forward(self, feat, adj):
        
        x = self.linear(feat)
        
        return x





class AttentionLayer(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  #[5,2]
        #print('alpha:', self.alpha)
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha      
    
  
  
    

   
             

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool, global_add_pool, global_max_pool, \
    GlobalAttention, JumpingKnowledge
from torch.nn import Sequential, Linear, ReLU, Conv2d, BatchNorm1d, LeakyReLU, Softplus, ELU
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import ClusterData, Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_undirected

import os
import time
from torch_sparse import SparseTensor, cat
from utils import build_virtual_edges

def get_conv_layer(name, in_channels, out_channels):
    if name.startswith("gcn"):
        return GCNConv(in_channels, out_channels, cached=True)
    elif name.startswith("sage"):
        return SAGEConv(in_channels, out_channels)
    elif name.startswith("gat"):
        return GATConv(in_channels, out_channels, heads=3, dropout=0.1)
    # elif name.startswith("gin"):
    #     return GINConv(
    #         Sequential(
    #             Linear(in_channels, hidden_channels),
    #             ReLU(),
    #             Linear(hidden_channels, out_channels)
    #         )
    #     )
    else:
        raise ValueError(f"{name} is not supported at this time!")

def get_activation(name):
    if name == "relu":
        return ReLU()
    elif name == "leaky":
        return LeakyReLU()
    elif name == "elu":
        return ELU()
    else:
        raise ValueError(f"{name} is unsupported at this time!")

def get_graph_pooling(name):
    if name == "sum":
        pool = global_add_pool
    elif name == "mean":
        pool = global_mean_pool
    elif name == "max":
        pool = global_max_pool
    else:
        raise ValueError(f"graph pooling - {name} is unsupported at this time!")
    return pool


class AdditiveAttention(nn.Module):
    """
     Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
     Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.

     Args:
         hidden_dim (int): dimesion of hidden state vector

     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.

     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.

     Reference:
         - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query, key, value):
        # query: (1, D)
        # key, values: (N, D)
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        # import ipdb; ipdb.set_trace()
        context = torch.mm(attn.unsqueeze(0), value)
        return context, attn

    
class VNTransmitterUnit(nn.Module):
    """
    It calculates message from graph node to virtual node.
    """
    def __init__(self, in_dim, out_dim, dropout_ratio=-1, activation=F.tanh):
        super(VNTransmitterUnit, self).__init__()
        self.attn = AdditiveAttention(in_dim)
        self.dropout_ratio = dropout_ratio
        if activation == "tanh":
            self.activation = F.tanh
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "ReLU":
            self.activation = F.relu
        elif activation == "None":
            self.activation = lambda x: x
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, h, g, vn_index):
        '''
        h: (N, D)
        g: (C, D)
        vn_index: (C, N)
        '''
        # import ipdb; ipdb.set_trace()
        cluster_num = g.shape[0]
        g_list = []
        for v in range(cluster_num):
            cluster_h = h[vn_index[v]]
            _, attn= self.attn(g[v], cluster_h, cluster_h)
            if self.dropout_ratio > 0.0:
                attn = F.dropout(attn, self.dropout_ratio, self.training)
            g_v = torch.mm(attn.unsqueeze(0), cluster_h)
            g_list.append(g_v)
        h_trans = torch.cat(g_list, dim=0)
        h_trans = self.W(h_trans)
        h_trans = self.activation(h_trans)
        return h_trans

class GNTransmitterUnit(nn.Module):
    """
    It calculates message from virtual node to graph node.
    """
    def __init__(self, hidden_dim_super=16, hidden_dim=16):
        super(GNTransmitterUnit, self).__init__()
        self.F_super = nn.Linear(hidden_dim_super, hidden_dim)
        
    def forward(self, g):
        """
        g: (C, I)
        """
        g_trans = self.F_super(g) # (C, D)
        g_trans = F.tanh(g_trans)
        return g_trans
    

class WarpGateUnit(nn.Module):
    """
    It computes gated-sum mixing `merged` feature from normal node feature `h`
    and super node feature `g`,
    """
    def __init__(self, hidden_dim=16, dropout_ratio=-1, activation=F.sigmoid):
        super(WarpGateUnit, self).__init__()
        self.H = nn.Linear(hidden_dim, hidden_dim)
        self.G = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_ratio = dropout_ratio
        self.activation = activation

    def forward(self, h, g, vn_index=None):
        # h: (N, D)
        # g: (C, D)
        # vn_index: (N, 2)
        # import ipdb; ipdb.set_trace()
        if vn_index is not None:
            g = torch.index_select(g, 0, vn_index[:, 1])
        z = self.H(h) + self.G(g)
        if self.dropout_ratio > 0.0:
            z = F.dropout(z, self.dropout_ratio)
        z = self.activation(z)
        merged = (1 - z) * h + z * g
        return merged
    
class VNGNN(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout, 
                 num_nodes, 
                 edge_index,
                 model, # "sage", "gcn"
                 dataset = "arxiv",
                 cluster_method="full", # ""metis", "full", "random"
                 use_virtual=False, 
                 num_clusters=0, 
                 JK=False,  
                 mode="cat",
                 sparse_ratio=1.0,
                 use_bn=1,
                 mlp_share=False,
                 args=None):
        
        super().__init__()
        
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.edge_index = edge_index
        self.dropout = dropout
        self.num_clusters = num_clusters
     
        self.cluster_method = cluster_method
        self.JK = JK 
        self.use_virtual = use_virtual
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.use_bn = use_bn

        if self.use_virtual:
            
            self.vn_index, self.cluster_adj = build_virtual_edges(edge_index, num_clusters, dataset, sparse_ratio=sparse_ratio, cluster_method=cluster_method)
            self.virtual_nodes = nn.Embedding(num_clusters, in_channels)
            channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels] # (layer + 1, )
            self.vn_trans = nn.ModuleList() # layer
            self.vn_update = nn.ModuleList() # layer - 1
            self.vn_merge = nn.ModuleList()

            self.gn_merge = nn.ModuleList()
            self.gn_trans = nn.ModuleList()
            for i in range(num_layers):
                self.gn_trans.append(GNTransmitterUnit(channels[i], channels[i+1]))
                self.gn_merge.append(WarpGateUnit(channels[i+1], args.merge_dropout))
            for i in range(num_layers - 1):
                # self.vn_update.append(nn.Linear(channels[i], channels[i+1]))
                self.vn_update.append(get_conv_layer(model, channels[i], channels[i+1]))
                self.vn_trans.append(VNTransmitterUnit(channels[i], channels[i+1], dropout_ratio=args.attn_dropout,activation=args.vntran_act)) 
                self.vn_merge.append(WarpGateUnit(channels[i+1], args.merge_dropout))


            # import ipdb; ipdb.set_trace()
        if model == "gat":
            # import ipdb; ipdb.set_trace()
            for i in range(num_layers):
                in_hidden = hidden_channels if i > 0 else in_channels
                out_hidden = hidden_channels // 3 if i < num_layers - 1 else out_channels
                num_heads = 3 if i < num_layers - 1 else 1
             
                self.convs.append(
                    GATConv(
                        in_hidden,
                        out_hidden ,
                        heads=num_heads,
                        attn_drop=0.1
                    )
                )
                if i < num_layers - 1:
                    self.bns.append(nn.BatchNorm1d(out_hidden * 3))
                    
        else:
            # graph node convoluntional layers
            self.convs.append(get_conv_layer(model, in_channels, hidden_channels))

            self.bns.append(BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(
                    get_conv_layer(model, hidden_channels, hidden_channels))
            
                self.bns.append(BatchNorm1d(hidden_channels ))
            

        # JK net
        if self.JK:
            self.convs.append(
                get_conv_layer(model, hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
            self.jump = JumpingKnowledge(mode=mode, channels=hidden_channels, num_layers=num_layers)
            if mode == 'cat':
                self.lin = Linear(num_layers * hidden_channels, out_channels)
            else:
                self.lin = Linear(hidden_channels, out_channels)
        else:
            if model != "gat":
                self.convs.append(
                    get_conv_layer(model, hidden_channels, out_channels))

        
        print("Model modules:\n", self)
        # import ipdb; ipdb.set_trace()


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.use_virtual:
            # self.vns.reset_parameters() # set the initial virtual node embedding to 0.
            nn.init.zeros_(self.virtual_nodes.weight.data)

    def forward(self, data, batch_train=False):
        """
        x:              [# of nodes, # of features]
        adj_t:          [# of nodes, # of nodes]
        virtual_node:   [# of virtual nodes, # of features]
        """
        # import ipdb; ipdb.set_trace()
        x = data.x
        adj_t = data.edge_index
        if self.use_virtual:
            vn_indices = torch.nonzero(self.vn_index.T).to(x.device)
            vx = self.virtual_nodes.weight
        # import ipdb; ipdb.set_trace()
        embs = [x] 
        # virtual_emb = [self.virtual_node.weight]
        h = x
        for layer in range(self.num_layers):
            # GNUpdate, graph node update, （1）
            h_hat = self.convs[layer](embs[layer], adj_t)  # (N, H)/(N, O)

            if self.use_virtual:
                if layer != self.num_layers - 1:
                    # VNUpdate, vraph node update, cross-cluster
                    g_hat = self.vn_update[layer](vx, self.cluster_adj.to(x.device)) # (5), (C, H)

                    # VNTrans, graph -> virtual node
                    g_trans = self.vn_trans[layer](embs[layer], vx, self.vn_index) # (6), (C, H)
                
                 # GNTrans, virtual node -> graph node
                h_trans = self.gn_trans[layer](vx)  # (2), (C, H)
                # GNMerge
                h = self.gn_merge[layer](h_hat, h_trans, vn_indices) # (3), (N, H)

                if layer != self.num_layers - 1:
                    # VNMerge
                    vx = self.vn_merge[layer](g_trans, g_hat) # (C, H)
                
            if layer != self.num_layers - 1 or self.JK:   
                h = self.bns[layer](h)             
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            embs.append(h)
                
        
        if self.JK:
            emb = self.jump(embs[1:])
            emb = self.lin(emb)
        else:
            emb = embs[-1]

        return emb.log_softmax(dim=-1)
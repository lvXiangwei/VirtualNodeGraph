from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool, global_add_pool, global_max_pool, \
    GlobalAttention, JumpingKnowledge
from torch.nn import Sequential, Linear, ReLU, Conv2d, BatchNorm1d, LeakyReLU, Softplus, ELU
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_undirected
from torch_geometric.loader import ClusterData
import os
import time
from torch_sparse import SparseTensor, cat
from utils import build_virtual_edges

def get_conv_layer(name, in_channels, out_channels, head=3):
    if name.startswith("gcn"):
        return GCNConv(in_channels, out_channels, cached=True)
    elif name.startswith("sage"):
        return SAGEConv(in_channels, out_channels)
    elif name.startswith("gat"):
        return GATConv(in_channels, out_channels// head, heads=head, dropout=0.1)
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

from layers import AdditiveAttention

class VirtualNode(nn.Module):
    def __init__(self, in_feats, hidden_feats, layers, vn_index, model = "gcn", num=1, edge_index=None, dropout=0.5, residual=False):
        super(VirtualNode, self).__init__()
        self.dropout = dropout
        # Add residual connection or not
        self.residual = residual
        self.vn_index = vn_index
        self.num_layer = layers


        self.vn_emb = nn.Embedding(num, in_feats)
        self.mlp_vns = nn.ModuleList()
        self.vn_convs = nn.ModuleList()
        # MLP to transform virtual node at every layer
        
        self.mlp_vns.append(nn.Sequential(
            nn.Linear(in_feats, 2 * hidden_feats),
            nn.BatchNorm1d(2 * hidden_feats),
            nn.ReLU(),
            nn.Linear(2 * hidden_feats, hidden_feats),
            nn.BatchNorm1d(hidden_feats),
            nn.ReLU())
        )
        
        
        for _ in range(layers - 2):
            self.mlp_vns.append(nn.Sequential(
                    nn.Linear(hidden_feats, 2 * hidden_feats),
                    nn.BatchNorm1d(2 * hidden_feats),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_feats, hidden_feats),
                    nn.BatchNorm1d(hidden_feats),
                    nn.ReLU())
            )
    
            self.vn_convs.append(get_conv_layer('gat', hidden_feats, hidden_feats, head=2))
        
        self.vn_convs.append(get_conv_layer('gat', hidden_feats, hidden_feats, head=2))
        
        self.virtual_num = num
        self.vn_edge_index = edge_index

        ### TODO, pooling attention
    
        self.attn_lst = nn.ModuleList([AdditiveAttention(in_feats),
            AdditiveAttention(hidden_feats)])
        
        self.reset_parameters()

    def reset_parameters(self):
        for c in self.vn_convs:
            c.reset_parameters()
        for c in self.mlp_vns.children():
           if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.vn_emb.weight.data, 0)

    def update_node_emb(self, layer, x, vn_indices, vx=None):
        r""" Add message from virtual nodes to graph nodes.
        Args:
            x (torch.Tensor): The input node feature.
            vn_indices: (# of cluster, # of node)
            vx (torch.Tensor, optional): Optional virtual node embedding.

        Returns:
            (torch.Tensor): The output node feature.
            (torch.Tensor): The output virtual node embedding.
        """
        # Virtual node embeddings for graphs
        # import ipdb; ipdb.set_trace()
        if vx is None:
            vx = self.vn_emb.weight
        # import ipdb; ipdb.set_trace()
        # Add message from virtual nodes to graph nodes
        if self.virtual_num > 1:
            # message passing between virtual node
            if layer > 0: # 第一层不需要更新(if layer == self.num_layer - 1: #只在最后一层更新)
                vx = self.vn_convs[layer - 1](vx, self.vn_edge_index.to(x.device))
            select_vns = torch.index_select(vx, 0, vn_indices[:, 1])
            h = x + select_vns
        else:
            h = x + vx
        return h, vx

    def update_vn_emb(self, layer, x, vx):
        r""" Add message from graph nodes to virtual node.
        Args:
            x (torch.Tensor): The input node feature.
            batch (LongTensor): Batch vector, which assigns each node to a specific example.
            vx (torch.Tensor): Optional virtual node embedding.

        Returns:
            (torch.Tensor): The output virtual node embedding.
        """
        # import ipdb; ipdb.set_trace()
        if self.virtual_num > 1:
            vx_tmp_list = []
            for v in range(self.virtual_num):
                # if layer == 0:
                #     vx_temp = global_add_pool(x[self.vn_index[v]], torch.zeros(1, dtype=torch.int64, device=x.device))
                # else:

                # ### TODO
                if layer == 0:
                    attn_layer = self.attn_lst[0]
                else:
                    attn_layer = self.attn_lst[1]
                cluster_h = x[self.vn_index[v]]
                _, attn= attn_layer(vx[v], cluster_h, cluster_h)
                if self.dropout > 0.0:
                    attn = F.dropout(attn, 0.1, self.training)
                vx_temp = torch.mm(attn.unsqueeze(0), cluster_h)
                vx_tmp_list.append(vx_temp)

            vx_temp = torch.cat(vx_tmp_list, dim=0) + vx
        else:
            vx_temp = global_add_pool(x, torch.zeros(1, dtype=torch.int64, device=x.device)) + vx

        # transform virtual nodes using MLP
        vx_temp = self.mlp_vns[layer](vx_temp)

        if self.residual:
            vx = vx + F.dropout(
                vx_temp, self.dropout, training=self.training)
        else:
            vx = F.dropout(
                vx_temp, self.dropout, training=self.training)

        return vx


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
                 use_bn=1, **kwargs):
        
        super().__init__()
        
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.edge_index = edge_index
        self.dropout = dropout
        self.num_clusters = num_clusters
        # import ipdb; ipdb.set_trace()
        self.cluster_method = cluster_method
        self.JK = JK 
        self.use_virtual = use_virtual
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.use_bn = use_bn


        if self.use_virtual:
            
            self.vn_index, self.cluster_adj = build_virtual_edges(edge_index, num_clusters, dataset, sparse_ratio=sparse_ratio, cluster_method=cluster_method)
            self.vns = VirtualNode(in_channels, hidden_channels, num_layers, self.vn_index, model, num_clusters, self.cluster_adj, self.dropout)

        if model == "gat":
            # import ipdb; ipdb.set_trace()
            for i in range(num_layers):
                in_hidden = hidden_channels if i > 0 else in_channels
                out_hidden = hidden_channels // 2 if i < num_layers - 1 else out_channels
                num_heads = 2 if i < num_layers - 1 else 1
             
                self.convs.append(
                    GATConv(
                        in_hidden,
                        out_hidden ,
                        heads=num_heads,
                        attn_drop=0.1
                    )
                )
                if i < num_layers - 1 and self.use_bn:
                    self.bns.append(nn.BatchNorm1d(out_hidden * 2))
                    
        else:
            # graph node convoluntional layers
            self.convs.append(get_conv_layer(model, in_channels, hidden_channels))
            if self.use_bn:
                self.bns.append(BatchNorm1d(hidden_channels))

            for _ in range(num_layers - 2):
                self.convs.append(
                    get_conv_layer(model, hidden_channels, hidden_channels))
                if self.use_bn:
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


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.use_virtual:
            self.vns.reset_parameters() # set the initial virtual node embedding to 0.
    
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
        # import ipdb; ipdb.set_trace()
        embs = [x]
        # virtual_emb = [self.virtual_node.weight]
        h = x
        for layer in range(self.num_layers):
            ### 0. Message passing among virtual nodes
            # virtual_node = self.convs_virtual[layer](virtual_emb[layer], self.cluster_adj.to(x.device))
            if self.use_virtual:
                if layer == 0:
                    vx = None
                h, vx = self.vns.update_node_emb(layer, h, vn_indices, vx)
            # import ipdb; ipdb.set_trace()
            ### 2. Message passing among graph nodes
            h = self.convs[layer](h, adj_t)  # GCN layer
            
            if layer != self.num_layers - 1 or self.JK:   
                if self.use_bn:
                    h = self.bns[layer](h)             
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            embs.append(h)

            if self.use_virtual:
                if layer != self.num_layers - 1 :
                    vx = self.vns.update_vn_emb(layer, embs[layer], vx)

        if self.JK:
            emb = self.jump(embs[1:])
            emb = self.lin(emb)
        else:
            emb = embs[-1]

        return emb

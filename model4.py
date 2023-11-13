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

def get_vn_index(name, num_ns, num_vns, num_vns_conn, edge_index, save_dir):
    if name == "full":
        idx = torch.ones(num_vns, num_ns)
    elif name == "random":
        idx = torch.zeros(num_vns, num_ns)
        for i in range(num_ns):
            rand_indices = torch.randperm(num_vns)[:num_vns_conn]
            idx[rand_indices, i] = 1
    elif name == "random-f" or name == "diffpool" or name == "random-e":
        return None
    elif "metis" in name:
        if not os.path.exists(save_dir):
            print("Clutsering...")
            start = time.time()
            clu = ClusterData(Data(edge_index=edge_index), num_parts=num_vns)
            idx = torch.zeros(num_vns, num_ns)
            for i in range(num_vns):
                idx[i][clu.perm[clu.partptr[i]:clu.partptr[i+1]]] = 1
            end = time.time()
            print(f"Cluster done, cost {end - start}")
            torch.save(idx, save_dir)
        else:
            idx = torch.load(save_dir)
            
    else:
        raise ValueError(f"{name} is unsupported at this time!")
    return idx == 1


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

    
class GNTransmitterUnit(nn.Module):
    """
    It calculates message from normal node to super node.
    """
    def __init__(self, in_dim, out_dim, dropout_ratio=-1, activation=F.tanh):
        super(GNTransmitterUnit, self).__init__()
        self.attn = AdditiveAttention(in_dim)
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, h, g, vn_index):
        '''
        h: (N, D)
        g: (C, D)
        vn_index: (C, N)
        '''
        cluster_num = g.shape[0]
        g_list = []
        for v in range(cluster_num):
            clutser_h = h[vn_index[v]]
            _, attn= self.attn(g[v], clutser_h, clutser_h)
            if self.dropout_ratio > 0.0:
                attn = F.dropout(attn, self.dropout_ratio)
            # import ipdb; ipdb.set_trace()
            g_v = torch.mm(attn.unsqueeze(0), clutser_h)
            g_list.append(g_v)
        h_trans = torch.cat(g_list, dim=0)
        h_trans = self.W(h_trans)
        h_trans = self.activation(h_trans)
        return h_trans


class VNTransmitterUnit(nn.Module):
    """
    It calculates message from super node to normal node.
    """
    def __init__(self, hidden_dim_super=16, hidden_dim=16, dropout_ratio=-1):
        super(VNTransmitterUnit, self).__init__()
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
        
        if vn_index is not None:
            # import ipdb; ipdb.set_trace()
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
                 clutser_method="full", # ""metis", "full", "random"
                 use_virtual=False, 
                 num_clusters=0, 
                 JK=False,  
                 mode="cat",
                 sparse_ratio=1.0,
                 use_bn=1):
        
        super().__init__()
        
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.edge_index = edge_index
        self.dropout = dropout
        self.num_clusters = num_clusters
        self.clutser_method = clutser_method
        self.JK = JK 
        self.use_virtual = use_virtual
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        

        if self.use_virtual:
            self.vn_index, self.cluster_adj = self.build_virtual_edges(sparse_ratio=sparse_ratio)
            self.virtual_nodes = nn.Embedding(num_clusters, in_channels)
            channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels] # (layer + 1, )
            self.vntrans = nn.ModuleList() # layer
            self.update_gn = nn.ModuleList() # layer - 1
            self.merge_gn = nn.ModuleList()
            self.merge_vn = nn.ModuleList()
            self.gntrans = nn.ModuleList()
            # import ipdb; ipdb.set_trace()
            for i in range(num_layers):
                self.vntrans.append(VNTransmitterUnit(channels[i], channels[i+1])) 
                self.merge_gn.append(WarpGateUnit(channels[i+1], 0.5))
            for i in range(num_layers - 1):
                self.update_gn.append(nn.Linear(channels[i], channels[i+1]))
                self.gntrans.append(GNTransmitterUnit(channels[i], channels[i+1], 0.5))
                self.merge_vn.append(WarpGateUnit(channels[i+1], 0.5))
            

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

    def build_virtual_edges(self, sparse_ratio=1.0):
        vn_index = get_vn_index(
            self.clutser_method, 
            self.num_nodes, 
            self.num_clusters, 
            1, 
            self.edge_index, 
            save_dir=f"cluster_data/{self.dataset}_clutser_{self.clutser_method}_{self.num_clusters}.pt") # (C, N), index[i] specifies which nodes are connected to VN i
        self.vn_index = vn_index
        cluster_adj = self.build_cluster_edges(self.edge_index, f"cluster_data/{self.dataset}_cluster_adj_{self.num_clusters}.pt")
        cluster_adj = self.adj_sparse(cluster_adj, sparse_ratio=sparse_ratio)
        
        return vn_index, cluster_adj 
    
    def adj_sparse(self, matrix, sparse_ratio=1.0):
        """
        matrix: cluster[i][j]: # of graph edges between cluster i and cluster i
        threshold: each cluster keeps "threshold" ratio of edges 
        """
        row_threshold = torch.topk(matrix, int(sparse_ratio * matrix.size(1)), dim=1, largest=True, sorted=False)[0][:, -1]
        # 将矩阵中小于阈值的元素置为0，大于等于阈值的元素置为1
        adj = (matrix >= row_threshold.view(-1, 1)).int()
        coo = coo_matrix(adj)
        edge_index = from_scipy_sparse_matrix(coo)[0]
        edge_index = to_undirected(edge_index)
        print(f"cluster edges: {edge_index.shape[1]}, total cluster edges:{matrix.shape[0] * (matrix.shape[0] - 1)}")
        return edge_index
     
    def build_cluster_edges(self, edge_index, save_dir):
        if os.path.exists(save_dir):
            cluster_adj = torch.load(save_dir)
            print(f"loading cluster edges from: {save_dir}")
        else:
            parts = self.num_clusters
            cluster_adj = torch.zeros((parts, parts), dtype=torch.long)
            vn_indices = torch.nonzero(self.vn_index.T)
            print("building cluster edges: ")
            start = time.time()
            for i, j in zip(edge_index[0], edge_index[1]):
                c_i, c_j = vn_indices[i][1], vn_indices[j][1]
                cluster_adj[c_i][c_j] += 1
            end = time.time()
            print(f"building cluster edges done, cost {end - start}")
            torch.save(cluster_adj, save_dir)

        return cluster_adj 

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.use_virtual:
            # self.vns.reset_parameters() # set the initial virtual node embedding to 0.
            nn.init.zeros_(self.virtual_nodes.weight.data)

    def forward(self, data):
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
            # graph node update
            h = self.convs[layer](h, adj_t)  # GCN layer #(N, D)
            if self.use_virtual:
                
                if layer != self.num_layers - 1:
                    # virtual node update
                    g_new = self.update_gn[layer](vx) # I(H) -> H

                    # graph -> virtual node
                    h_trans = self.gntrans[layer](embs[layer], vx, self.vn_index)  # 
                
                # virtual -> graph node
                g_trans = self.vntrans[layer](vx)
                h = self.merge_gn[layer](h, g_trans, vn_indices) 

                if layer != self.num_layers - 1:
                    # import ipdb; ipdb.set_trace()
                    vx = self.merge_vn[layer](h_trans, g_new)
                
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
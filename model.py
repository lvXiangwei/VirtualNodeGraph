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


class Attention(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def foward(self, graph_node, virtual_node):
        '''
        graph node: (N, D)
        virtual node: (C, D)
        '''
        N = graph_node.shape[0]
        C = virtual_node.shape[1]
        graph_node = torch.mm(graph_node, self.W)
        virtual_node = torch.mm(virtual_node, self.W)
        a_input = torch.cat([graph_node.repeat(1, C).view(C * N, -1), virtual_node.repeat(N, 1)], dim=1).view(N, -1, 2*graph_node.shape[1]) # (N, C, 2*D)
        attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)).view(N, C)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, virtual_node) # (N, D)
        return h_prime


class VirtualNode(nn.Module):
    def __init__(self, in_feats, hidden_feats, layers, vn_index, model = "gcn", num=1, edge_index=None, dropout=0.5, residual=False):
        super(VirtualNode, self).__init__()
        self.dropout = dropout
        # Add residual connection or not
        self.residual = residual
        self.vn_index = vn_index
        self.num_layer = layers
        # Set the initial virtual node embedding to 0.
        self.vn_emb = nn.Embedding(num, in_feats)
        # nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # if in_feats == out_feats:
        #     self.linear = nn.Identity()
        # else:
        #     self.linear = nn.Linear(in_feats, out_feats)

        # MLP to transform virtual node at every layer
        self.mlp_vns = nn.ModuleList()
        self.mlp_vns.append(nn.Sequential(
            # nn.Linear(in_feats, 2 * hidden_feats),
            # nn.BatchNorm1d(2 * hidden_feats),
            # nn.ReLU(),
            # nn.Linear(2 * hidden_feats, hidden_feats),
            # nn.BatchNorm1d(hidden_feats),
            # nn.ReLU())
            nn.Linear(in_feats, hidden_feats),
            nn.LayerNorm(hidden_feats),
            nn.ReLU()
            )
        )

        # message passing between virtual node
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(get_conv_layer(model, in_feats, in_feats))
        # self.bns.append(nn.BatchNorm1d(in_feats))
    

        for _ in range(layers - 2):
            self.mlp_vns.append(nn.Sequential(
                    # nn.Linear(hidden_feats, 2 * hidden_feats),
                    # nn.BatchNorm1d(2 * hidden_feats),
                    # nn.ReLU(),
                    # nn.Linear(2 * hidden_feats, hidden_feats),
                    # nn.BatchNorm1d(hidden_feats),
                    # nn.ReLU())
                    nn.Linear(hidden_feats, hidden_feats),
                    nn.LayerNorm(hidden_feats),
                    nn.ReLU()
                )
            )
            self.convs.append(get_conv_layer(model, hidden_feats, hidden_feats))
            # self.bns.append(nn.BatchNorm1d(hidden_feats)),
        self.convs.append(get_conv_layer(model, hidden_feats, hidden_feats))
        
        self.virtual_num = num
        self.virtual_edge_index = edge_index
 
        self.reset_parameters()

    def reset_parameters(self):
        # if not isinstance(self.linear, nn.Identity):
        #     self.linear.reset_parameters()
        for c in self.convs:
            c.reset_parameters()
        for c in self.bns:
            c.reset_parameters()
        for c in self.mlp_vns.children():
           if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.vn_emb.weight.data, 0)

    def update_node_emb(self, layer, x, vn_indices, vx=None):
        r""" Add message from virtual nodes to graph nodes.
        Args:
            x (torch.Tensor): The input node feature.
            vn_indices: (# of clutser, # of node)
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
            # import ipdb; ipdb.set_trace()
            if layer == self.num_layer - 1:
                vx = self.convs[layer](vx, self.virtual_edge_index.to(x.device))
                # pass 
            # vx = self.bns[layer](vx)
            # vx = F.relu(vx)
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

        # Add message from graph nodes to virtual nodes
        # vx = self.linear(vx)
        # import ipdb; ipdb.set_trace()
        if self.virtual_num > 1:
            vx_tmp_list = []

            for v in range(self.virtual_num):
                vx_temp = global_add_pool(x[self.vn_index[v]], torch.zeros(1, dtype=torch.int64, device=x.device))
                vx_tmp_list.append(vx_temp)
            # import ipdb; ipdb.set_trace()
            vx_temp = torch.cat(vx_tmp_list, dim=0) + vx
        else:
            # import ipdb; ipdb.set_trace()
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
                 clutser_method="full", # ""metis", "full", "random"
                 use_virtual=False, 
                 num_clusters=0, 
                 JK=False,  
                 mode="cat",
                 sparse_ratio=1.0):
        
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
            self.vns = VirtualNode(in_channels, hidden_channels, num_layers, self.vn_index, model, num_clusters, self.cluster_adj, self.dropout)

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
            self.vns.reset_parameters() # set the initial virtual node embedding to 0.
    
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

        return emb.log_softmax(dim=-1)
    

class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


# class GAT(nn.Module):
#     def __init__(
#         self,
#         in_feats,
#         n_classes,
#         n_hidden,
#         n_layers,
#         n_heads,
#         activation,
#         dropout=0.0,
#         input_drop=0.0,
#         attn_drop=0.0,
#         edge_drop=0.0,
#         use_attn_dst=True,
#         use_symmetric_norm=False,
#     ):
#         super().__init__()
#         self.in_feats = in_feats
#         self.n_hidden = n_hidden
#         self.n_classes = n_classes
#         self.n_layers = n_layers
#         self.num_heads = n_heads

#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()

#         for i in range(n_layers):
#             in_hidden = n_heads * n_hidden if i > 0 else in_feats
#             out_hidden = n_hidden if i < n_layers - 1 else n_classes
#             num_heads = n_heads if i < n_layers - 1 else 1
#             out_channels = n_heads

#             self.convs.append(
#                 GATConv(
#                     in_hidden,
#                     out_hidden,
#                     num_heads=num_heads,
#                     attn_drop=attn_drop,
#                     edge_drop=edge_drop,
#                     use_attn_dst=use_attn_dst,
#                     use_symmetric_norm=use_symmetric_norm,
#                     residual=True,
#                 )
#             )

#             if i < n_layers - 1:
#                 self.norms.append(nn.BatchNorm1d(out_channels * out_hidden))

#         self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

#         self.input_drop = nn.Dropout(input_drop)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = activation

#     def forward(self, graph, feat):
#         h = feat
#         h = self.input_drop(h)

#         for i in range(self.n_layers):
#             conv = self.convs[i](graph, h)

#             h = conv

#             if i < self.n_layers - 1:
#                 h = h.flatten(1)
#                 h = self.norms[i](h)
#                 h = self.activation(h, inplace=True)
#                 h = self.dropout(h)

#         h = h.mean(1)
#         h = self.bias_last(h)

#         return h
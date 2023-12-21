from prettytable import PrettyTable
import torch 
import random
import numpy as np
import yaml

class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def seed_everything(seed=42):
    # Set the seed for random module
    random.seed(seed)

    # Set the seed for numpy module
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure reproducibility with cudnn (if installed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

import torch_sparse
def build_virtual_edges(edge_index, num_clusters, dataset="arxiv", sparse_ratio=1.0, cluster_method="metis"):
    # import ipdb; ipdb.set_trace()
    if type(edge_index) == torch_sparse.tensor.SparseTensor:
        row, col, _ = edge_index.coo()
        edge_index = torch.stack([row, col], dim=0)
        
    num_nodes = edge_index.max() + 1
    vn_index = get_vn_index(
        cluster_method, 
        num_nodes, 
        num_clusters, 
        1, 
        edge_index, 
        save_dir=f"cluster_data/{dataset}/{dataset}_clutser_{cluster_method}_{num_clusters}.pt") # (C, N), index[i] specifies which nodes are connected to VN i

    cluster_adj = build_cluster_edges(edge_index, num_clusters, vn_index, f"cluster_data/{dataset}/{dataset}_cluster_adj_{num_clusters}.pt")
    cluster_adj = adj_sparse(cluster_adj, sparse_ratio=sparse_ratio)
    
    return vn_index, cluster_adj 

import os
from torch_geometric.data import ClusterData, Data
import time

def get_vn_index(name, num_ns, num_vns, num_vns_conn, edge_index, save_dir):
    if name == "full":
        idx = torch.ones(num_vns, num_ns)
    elif name == "random":
        # import ipdb; ipdb.set_trace()
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

def build_cluster_edges(edge_index, num_clusters, vn_index, save_dir):
    if os.path.exists(save_dir):
        cluster_adj = torch.load(save_dir)
        print(f"loading cluster edges from: {save_dir}")
    else:
        parts = num_clusters
        cluster_adj = torch.zeros((parts, parts), dtype=torch.long)
        vn_indices = torch.nonzero(vn_index.T)
        print("building cluster edges: ")
        start = time.time()
        for i, j in zip(edge_index[0], edge_index[1]):
            c_i, c_j = vn_indices[i][1], vn_indices[j][1]
            cluster_adj[c_i][c_j] += 1
        end = time.time()
        print(f"building cluster edges done, cost {end - start}")
        torch.save(cluster_adj, save_dir)

    return cluster_adj 

from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_undirected

def adj_sparse(matrix, sparse_ratio=1.0):
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
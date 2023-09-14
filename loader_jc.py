from typing import List, Union, Tuple
import logging
from math import ceil

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
from torch_sparse import SparseTensor

from dataloaders.utils import topk_ppr_matrix
from torch_geometric.loader.base import BaseDataLoader
import ipdb
import time
from torch_sparse import SparseTensor, cat
import copy
import os.path as osp
from typing import Optional
import sys

class shadowloader(BaseDataLoader):

    def __init__(self, graph: Data,
                 output_indices: torch.LongTensor,
                 return_edge_index_type: str,
                 num_auxiliary_node_per_output: int,
                 alpha: float = 0.2,
                 eps: float = 1.e-4,
                 batch_size: int = 1,
                 **kwargs):

        self.out_aux_pairs = []

        assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes), "Assume the graph to be undirected"
        self.output_indices = output_indices
        assert return_edge_index_type in ['adj', 'edge_index']
        self.data = graph
        self.return_edge_index_type = return_edge_index_type
        self.num_auxiliary_node_per_output = num_auxiliary_node_per_output
        self.batch_size = batch_size
        self.alpha = alpha
        self.eps = eps

        self.original_graph = graph  # need to cache the original graph
        self.adj = None
        self.create_node_wise_loader(graph)

        # super().__init__(self.out_aux_pairs,
        #                  batch_size=batch_size, **kwargs)
        # ipdb.set_trace()
        super().__init__(self.out_aux_pairs,
                         batch_size=batch_size, collate_fn=self.__collate__,
                         **kwargs)

    @classmethod
    def indices_complete_check(cls,
                               loader: List[Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]],
                               output_indices: Union[torch.Tensor, np.ndarray]):
        if isinstance(output_indices, torch.Tensor):
            output_indices = output_indices.cpu().numpy()

        outs = []
        for out, aux in loader:
            if isinstance(out, torch.Tensor):
                out = out.cpu().numpy()
            if isinstance(aux, torch.Tensor):
                aux = aux.cpu().numpy()

            assert np.all(np.in1d(out, aux)), "Not all output nodes are in aux nodes!"
            outs.append(out)

        outs = np.sort(np.concatenate(outs))
        assert np.all(outs == np.sort(output_indices)), "Output nodes missing or duplicate!"

    def create_node_wise_loader(self, graph: Data):
        # logging.info("Start PPR calculation")
        t = time.time()
        print("Start PPR calculation")
        _, neighbors = topk_ppr_matrix(graph.edge_index,
                                       graph.num_nodes,
                                       self.alpha,
                                       self.eps,
                                       self.output_indices,
                                       self.num_auxiliary_node_per_output)

        for p, n in zip(self.output_indices.numpy(), neighbors):
            self.out_aux_pairs.append((np.array([p]), n))
        # ipdb.set_trace()
        print(f'time for PPR calculation: {time.time()-t}s')
        # ipdb.set_trace()
        self.indices_complete_check(self.out_aux_pairs, self.output_indices)

        # if self.return_edge_index_type == 'adj':
        #     adj = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(graph.num_nodes, graph.num_nodes))
        #     adj = self.normalize_adjmat(adj, normalization='rw')
        #     self.adj = adj

    # def __getitem__(self, idx):
    #     return self.out_aux_pairs[idx]

    def __len__(self):
        return len(self.out_aux_pairs)

    @property
    def loader_len(self):
        return ceil(len(self.out_aux_pairs) / self.batch_size)

    def __collate__(self, data_list):
        # ipdb.set_trace()
        # return self.data.subgraph(index)
        # ipdb.set_trace()
        out, aux = zip(*data_list)
        out = np.concatenate(out)
        aux = list(set(np.concatenate(aux)))
        mask = torch.from_numpy(np.in1d(aux, out))
        # ipdb.set_trace()
        subg = self.data.subgraph(torch.tensor(aux))
        subg.x = self.data.x[aux]
        subg.y = self.data.y[aux]
        subg.aux_indices = aux
        subg.target_mask = mask
        subg.batch_size = self.batch_size
        # new_subg = Data(x=self.data.x[aux],
        #                 y=self.data.y[aux],
        #                 edge_index=edge_index,
        #                 edge_attr=edge_attr,
        #                 target_mask=mask)
        return subg
        # out = np.concatenate(out)
        # aux = np.concatenate(aux)  # DO NOT UNION!
        # mask = torch.from_numpy(np.in1d(aux, out))
        # aux = torch.from_numpy(aux)
        #
        # subg = self.get_subgraph(aux,
        #                          self.original_graph,
        #                          self.return_edge_index_type,
        #                          self.adj,
        #                          output_node_mask=mask)
        # return subg


class clusterdata(torch.utils.data.Dataset):
    r"""Clusters/partitions a graph data object into multiple subgraphs, as
    motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
    and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
        save_dir (string, optional): If set, will save the partitioned data to
            the :obj:`save_dir` directory for faster re-use.
            (default: :obj:`None`)
        log (bool, optional): If set to :obj:`False`, will not log any
            progress. (default: :obj:`True`)
    """
    def __init__(self, data, num_parts: int, recursive: bool = False,
                 save_dir: Optional[str] = None, log: bool = True, split_idx=None, no_mask=False,
                 new_mask_name='train'):

        assert data.edge_index is not None

        self.num_parts = num_parts
        self.split_idx = split_idx
        self.no_mask = no_mask
        self.new_mask_name = new_mask_name
        # ipdb.set_trace()
        if len(data.y.size()) == 2 and data.y.sum() >= data.x.size(0):
            self.multi_labels = True
        else:
            self.multi_labels = False

        recursive_str = '_recursive' if recursive else ''
        filename = f'partition_{num_parts}{recursive_str}.pt'
        path = osp.join(save_dir or '', filename)
        if save_dir is not None and osp.exists(path):
            print(f'loaded METIS partitions... (num_parts: {num_parts})')
            adj, new_adj, partptr, perm = torch.load(path)
        else:
            if log:  # pragma: no cover
                print('Computing METIS partitioning...', file=sys.stderr)

            N, E = data.num_nodes, data.num_edges
            adj = SparseTensor(
                row=data.edge_index[0], col=data.edge_index[1],
                value=torch.arange(E, device=data.edge_index.device),
                sparse_sizes=(N, N))
            # ipdb.set_trace()
            adj, partptr, perm = adj.partition(num_parts, recursive)

            # if save_dir is not None:
            #     torch.save((adj, partptr, perm), path)

            # if log:  # pragma: no cover
            #     print('Done!', file=sys.stderr)

            # TODO: add coarse nodes
            coarse_nodes = torch.arange(N, N+num_parts)
            new_row, new_col = [], []
            for i in range(len(partptr)-1): # fine nodes to coarse nodes
                start, end = partptr[i], partptr[i+1]
                new_row += list(range(start, end))
                new_col += [i+N for _ in range(start, end)]
                new_row += [i+N for _ in range(start, end)] # undirected
                new_col += list(range(start, end))
            cnt = 0
            t1 = time.time()
            for i in range(num_parts):
                for j in range(num_parts):
                    if i == j:
                        continue
                    # ipdb.set_trace()
                    start1, end1 = partptr[i], partptr[i+1]
                    start2, end2 = partptr[j], partptr[j+1]
                    # all_idx = torch.cat([torch.arange(start1, end1), torch.arange(start2, end2)])
                    part1_idx = torch.arange(start1, end1)
                    part2_idx = torch.arange(start2, end2)
                    nnz = adj.index_select(0, part1_idx).index_select(1, part2_idx).nnz()
                    if nnz != 0:
                        new_row += [i+N]
                        new_col += [j+N]
                        cnt += 1
            t2 = time.time()
            print(f'cnt: {cnt}, time: {t2-t1}s, full graph cnt: {num_parts*(num_parts-1)}')
            num_new_E = len(new_row)
            new_row = torch.cat((adj.storage.row(), torch.tensor(new_row)))
            new_col = torch.cat((adj.storage.col(), torch.tensor(new_col)))
            new_E_val = torch.arange(E+num_new_E)
            # ipdb.set_trace()
            new_adj = SparseTensor(
                row=new_row, col=new_col,
                value=new_E_val,
                sparse_sizes=(N+num_parts, N+num_parts))

            if save_dir is not None:
                torch.save((adj, new_adj, partptr, perm), path)
            if log:  # pragma: no cover
                print('Done!', file=sys.stderr)

        # ipdb.set_trace()
        self.ori_data = self.__permute_data__(data, perm, adj)
        self.partptr = partptr
        self.perm = perm
        # ipdb.set_trace()
        # add features for coarse nodes
        self.data = self.__add_coarse_nodes__(self.ori_data, num_parts, new_adj)
        # ipdb.set_trace()

    def __permute_data__(self, data, node_idx, adj):
        data = copy.copy(data)
        N = data.num_nodes

        for key, item in data:
            if isinstance(item, torch.Tensor) and item.size(0) == N:
                data[key] = item[node_idx]

        data.edge_index = None
        data.adj = adj

        return data

    def __add_coarse_nodes__(self, data, num_parts, adj):
        data = copy.copy(data)
        ori_N = data.num_nodes
        data.num_nodes = data.num_nodes + num_parts
        # ipdb.set_trace()
        new_x = torch.zeros((num_parts, data.x.size(1)))
        if self.multi_labels:
            new_y = -torch.ones((num_parts, data.y.size(1)))
        else:
            new_y = -torch.ones(num_parts)
        new_mask = [False for _ in range(num_parts)]
        if not self.no_mask:
            for mask in ['train_mask', 'val_mask', 'test_mask']:
                if self.split_idx: # for ogbn dataset
                    mask_type = mask.split('_mask')[0]
                    if mask_type == 'val':
                        mask_type = 'valid'
                    tmp_mask = torch.tensor([False for _ in range(ori_N)])
                    tmp_mask[self.split_idx[mask_type]] = True
                    data.__setitem__(mask, tmp_mask)
                data[mask] = torch.cat((data[mask], torch.tensor(new_mask)))
                if not self.multi_labels:
                    data.y = data.y.view(-1)
        # else:
        #     ori_mask = torch.tensor([True for _ in range(ori_N)])
        #     data[self.new_mask_name] = torch.cat((ori_mask, torch.tensor(new_mask)))
        data.x = torch.cat((data.x, new_x), dim=0)
        data.y = torch.cat((data.y, new_y), dim=0)
        # ipdb.set_trace()
        row, col, _ = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        return data

    def __len__(self):
        return self.partptr.numel() - 1

    def __getitem__(self, idx):
        start = int(self.partptr[idx])
        length = int(self.partptr[idx + 1]) - start

        N, E = self.data.num_nodes, self.data.num_edges
        data = copy.copy(self.data)
        del data.num_nodes
        adj, data.adj = data.adj, None

        adj = adj.narrow(0, start, length).narrow(1, start, length)
        edge_idx = adj.storage.value()

        for key, item in data:
            if isinstance(item, torch.Tensor) and item.size(0) == N:
                data[key] = item.narrow(0, start, length)
            elif isinstance(item, torch.Tensor) and item.size(0) == E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        row, col, _ = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  data={self.data},\n'
                f'  num_parts={self.num_parts}\n'
                f')')

class clusterloader(torch.utils.data.DataLoader):
    r"""The data loader scheme from the `"Cluster-GCN: An Efficient Algorithm
    for Training Deep and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper which merges partioned subgraphs
    and their between-cluster links from a large-scale graph data object to
    form a mini-batch.

    .. note::

        Use :class:`~torch_geometric.loader.ClusterData` and
        :class:`~torch_geometric.loader.ClusterLoader` in conjunction to
        form mini-batches of clusters.
        For an example of using Cluster-GCN, see
        `examples/cluster_gcn_reddit.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py>`_ or
        `examples/cluster_gcn_ppi.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/cluster_gcn_ppi.py>`_.

    Args:
        cluster_data (torch_geometric.loader.ClusterData): The already
            partioned data object.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, cluster_data, **kwargs):
        self.cluster_data = cluster_data

        super().__init__(range(len(cluster_data)), collate_fn=self.__collate__,
                         **kwargs)

    def __collate__(self, batch):
        ipdb.set_trace()
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        N = self.cluster_data.data.num_nodes
        E = self.cluster_data.data.num_edges

        start = self.cluster_data.partptr[batch].tolist()
        end = self.cluster_data.partptr[batch + 1].tolist()
        node_idx = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])

        data = copy.copy(self.cluster_data.data)
        del data.num_nodes
        adj, data.adj = self.cluster_data.data.adj, None
        adj = cat([adj.narrow(0, s, e - s) for s, e in zip(start, end)], dim=0)
        adj = adj.index_select(1, node_idx)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in data:
            if isinstance(item, torch.Tensor) and item.size(0) == N:
                data[key] = item[node_idx]
            elif isinstance(item, torch.Tensor) and item.size(0) == E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        return data

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit, Reddit2
import numpy as np
import torch

from torch_geometric.utils import to_undirected, add_remaining_self_loops

def load_data(dataset_name: str,
              small_trainingset: float,
              pretransform):
    """

    :param dataset_name:
    :param small_trainingset:
    :param pretransform:
    :return:
    """
    if dataset_name.lower() in ['arxiv', 'products', 'papers100m']:
        dataset = PygNodePropPredDataset(name="ogbn-{:s}".format(dataset_name),
                                         root='./dataset',
                                         pre_transform=pretransform)
        split_idx = dataset.get_idx_split()
        graph = dataset[0]

        # TODO: add mask 
        N = graph.num_nodes
        for mask_type in ["train", "valid", "test"]:
            graph[f'{mask_type}_mask'] = torch.BoolTensor(N) * False 
            graph[f'{mask_type}_mask'][split_idx[mask_type]] = True

    elif dataset_name.lower().startswith('reddit'):
        if dataset_name == 'reddit2':
            dataset = Reddit2('./dataset/Reddit2', pre_transform=pretransform)
        elif dataset_name == 'reddit':
            dataset = Reddit('./dataset/Reddit', pre_transform=pretransform)
        else:
            raise ValueError
        graph = dataset[0]
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
        # graph.train_mask, graph.val_mask, graph.test_mask = None, None, None
    else:
        raise NotImplementedError

    train_indices = split_idx["train"].numpy()

    if small_trainingset < 1:
        np.random.seed(2021)
        train_indices = np.sort(np.random.choice(train_indices,
                                                 size=int(len(train_indices) * small_trainingset),
                                                 replace=False,
                                                 p=None))

    train_indices = torch.from_numpy(train_indices)

    val_indices = split_idx["valid"]
    test_indices = split_idx["test"]
    return graph, split_idx


class GraphPreprocess:
    def __init__(self,
                 self_loop: bool = True,
                 transform_to_undirected: bool = True):
        self.self_loop = self_loop
        self.to_undirected = transform_to_undirected

    def __call__(self, graph: Data):
        graph.y = graph.y.reshape(-1)
        graph.y = torch.nan_to_num(graph.y, nan=-1)
        graph.y = graph.y.to(torch.long)

        if self.self_loop:
            edge_index, _ = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
        else:
            edge_index = graph.edge_index

        if self.to_undirected:
            edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)

        graph.edge_index = edge_index
        return graph




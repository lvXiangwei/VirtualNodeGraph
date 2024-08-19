from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit, Reddit2
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Flickr, PPI
from numpy.random import default_rng

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
            dataset = Reddit2('./dataset/reddit2', pre_transform=pretransform)
        elif dataset_name == 'reddit':
            dataset = Reddit('./dataset/reddit', pre_transform=pretransform)
        else:
            raise ValueError
        graph = dataset[0]
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
        import ipdb; ipdb.set_trace()
        # graph.train_mask, graph.val_mask, graph.test_mask = None, None, None
    elif dataset_name.lower() in ["pubmed"]:
        dataset = Planetoid('./dataset', dataset_name, transform=pretransform)
        graph = dataset[0]
        graph.valid_mask = graph.val_mask
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
    elif dataset_name.lower() in ["flickr"]:
        dataset = Flickr('./dataset/flickr', transform=pretransform)
        graph = dataset[0]
        graph.valid_mask = graph.val_mask
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
    elif dataset_name.lower() in ["proteins"]:
        import torch_geometric.transforms as T
        dataset = PygNodePropPredDataset(
            name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
        graph = dataset[0]
        # import ipdb; ipdb.set_trace()
        # Move edge features to node features.
        graph.x = graph.adj_t.mean(dim=1)
        graph.adj_t.set_value_(None)

        # row, col, _ = graph.adj_t.coo()
        # graph.edge_index = torch.stack([row, col], dim=0)
      
        split_idx = dataset.get_idx_split()
        N = graph.num_nodes
        for mask_type in ["train", "valid", "test"]:
            graph[f'{mask_type}_mask'] = torch.BoolTensor(N) * False 
            graph[f'{mask_type}_mask'][split_idx[mask_type]] = True
    elif dataset_name.lower() in ["cora"]:
        import torch_geometric.transforms as T
        dataset = Planetoid('dataset', "Cora", transform=T.NormalizeFeatures())
        graph = dataset[0]
        graph['valid_mask'] = graph['val_mask']
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
    elif dataset_name.lower() in ["corasmod2"]:
        import torch_geometric.transforms as T
        from functools import partial
        path = "./dataset/PlanetoidSModK2"
        tf_list = [T.NormalizeFeatures(),
                    partial(sanitize_transductive_task, k=2, resample=True)]
        # k = 3
        # import ipdb; ipdb.set_trace()
        dataset = Planetoid(path, "Cora",
                            pre_transform=T.Compose(tf_list))
        print(f">> Using sanitized version with k={2}")
        graph = dataset[0]
        graph['valid_mask'] = graph['val_mask']
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
    elif dataset_name.lower() in ["pubmedsmod2"]:
        import torch_geometric.transforms as T
        from functools import partial
        path = "./dataset/PlanetoidSModK2"
        tf_list = [T.NormalizeFeatures(),
                    partial(sanitize_transductive_task, k=2, resample=True)]
        # k = 3
        # import ipdb; ipdb.set_trace()
        dataset = Planetoid(path, "Pubmed",
                            pre_transform=T.Compose(tf_list))
        print(f">> Using sanitized version with k={2}")
        graph = dataset[0]
        graph['valid_mask'] = graph['val_mask']
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
    elif dataset_name.lower() in ["corasmod2"]:
        import torch_geometric.transforms as T
        from functools import partial
        path = "./dataset/PlanetoidSModK2"
        tf_list = [T.NormalizeFeatures(),
                    partial(sanitize_transductive_task, k=2, resample=True)]
        # k = 3
        # import ipdb; ipdb.set_trace()
        dataset = Planetoid(path, "Cora",
                            pre_transform=T.Compose(tf_list))
        print(f">> Using sanitized version with k={2}")
        graph = dataset[0]
        graph['valid_mask'] = graph['val_mask']
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
    elif dataset_name.lower() in ["citeseersmod2"]:
        import torch_geometric.transforms as T
        from functools import partial
        path = "./dataset/PlanetoidSModK2"
        tf_list = [T.NormalizeFeatures(),
                    partial(sanitize_transductive_task, k=2, resample=True)]
        # k = 3
        # import ipdb; ipdb.set_trace()
        dataset = Planetoid(path, "CiteSeer",
                            pre_transform=T.Compose(tf_list))
        print(f">> Using sanitized version with k={2}")
        graph = dataset[0]
        graph['valid_mask'] = graph['val_mask']
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.val_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
        
    elif dataset_name.lower()[:7] in ["cluster"]:
        graph = torch.load(f"dataset/custom/{dataset_name}.pt")
        graph.y = graph.y.long()
        import ipdb; ipdb.set_trace()
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                     'valid': graph.valid_mask.nonzero().reshape(-1),
                     'test': graph.test_mask.nonzero().reshape(-1)}
    elif dataset_name.lower()[:7] in ["dblp"]:
        from torch_geometric.datasets import Planetoid, CitationFull
        import torch_geometric.transforms as T
        import os
        dataset = CitationFull('dataset', 'DBLP', transform=T.NormalizeFeatures())

        path = "./dataset/dblp/graph.pt"
        if os.path.exists(path):
            graph = torch.load(path)
        else:
            # dataset = Amazon('data', dataset_name, transform=T.NormalizeFeatures())
            graph = dataset[0]
            # graph = random_coauthor_amazon_splits(graph, dataset.num_classes)
            graph = split_ratio(graph)
            # graph.valid_mask = graph.val_mask
            torch.save(graph, path)
        print(graph)
        
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                    'valid': graph.valid_mask.nonzero().reshape(-1),
                    'test': graph.test_mask.nonzero().reshape(-1)}
        # import ipdb; ipdb.set_trace()
        
    elif dataset_name.lower()[:7] in ["cs", "physics"]:
        from torch_geometric.datasets import Coauthor
        import torch_geometric.transforms as T
        import os
        path = "./data/CS/graph.pt"
        if os.path.exists(path):
            graph = torch.load(path)
        else:
            dataset = Coauthor('data', dataset_name, transform=T.NormalizeFeatures())
            graph = dataset[0]
            # graph = random_coauthor_amazon_splits(graph, dataset.num_classes)
            # graph.valid_mask = graph.val_mask
            graph = split_ratio(graph)
            torch.save(graph, path)
        
        # import ipdb; ipdb.set_trace()
        print(graph)
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                    'valid': graph.valid_mask.nonzero().reshape(-1),
                    'test': graph.test_mask.nonzero().reshape(-1)}
        
    elif dataset_name.lower() in ["computers", "photo"]:
        from torch_geometric.datasets import Amazon
        import torch_geometric.transforms as T
        import os
        path = "./data/Computers/graph.pt"
        if os.path.exists(path):
            
            graph = torch.load(path)
        else:
            dataset = Amazon('data', dataset_name, transform=T.NormalizeFeatures())
            graph = dataset[0]
            # graph = random_coauthor_amazon_splits(graph, dataset.num_classes)
            graph = split_ratio(graph)
            # graph.valid_mask = graph.val_mask
            torch.save(graph, path)
        
        # import ipdb; ipdb.set_trace()
        print(graph)
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                    'valid': graph.valid_mask.nonzero().reshape(-1),
                    'test': graph.test_mask.nonzero().reshape(-1)}

    elif dataset_name.lower() in ["blogcatalog"]:
        from torch_geometric.datasets import Amazon, AttributedGraphDataset
        import torch_geometric.transforms as T
        import os
        path = "./data/blogcatalog/graph.pt"
        if os.path.exists(path):
            
            graph = torch.load(path)
        else:
            
            dataset = AttributedGraphDataset("data", "BlogCatalog", transform=T.NormalizeFeatures())
            graph = dataset[0]
            # import ipdb; ipdb.set_trace()
            
            # graph = random_coauthor_amazon_splits(graph, dataset.num_classes)
            graph = split_ratio(graph)
            # graph.valid_mask = graph.val_mask
            torch.save(graph, path)
        

        # import ipdb; ipdb.set_trace()
        print(graph)
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                    'valid': graph.valid_mask.nonzero().reshape(-1),
                    'test': graph.test_mask.nonzero().reshape(-1)}
    elif dataset_name.lower() in ["wikics"]:
        from torch_geometric.datasets import Amazon, AttributedGraphDataset, WikiCS
        import torch_geometric.transforms as T
        import os
        path = "./data/wikics/graph.pt"
        if os.path.exists(path):
            graph = torch.load(path)
        else:
            
            dataset = WikiCS("data/wikics", transform=T.NormalizeFeatures())
            graph = dataset[0]
            graph = dataset[0]
            graph['valid_mask'] = graph['val_mask']
            graph['train_mask'] = graph['train_mask'][:, 0]
            graph['valid_mask'] = graph['valid_mask'][:, 0]
            # import ipdb; ipdb.set_trace()
            split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                        'valid': graph.val_mask.nonzero().reshape(-1),
                        'test': graph.test_mask.nonzero().reshape(-1)}
        

        # import ipdb; ipdb.set_trace()
        # print(graph)
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                    'valid': graph.valid_mask.nonzero().reshape(-1),
                    'test': graph.test_mask.nonzero().reshape(-1)}
    elif dataset_name.lower() in ["corafull"]:
        from torch_geometric.datasets import Amazon, AttributedGraphDataset, CoraFull
        import torch_geometric.transforms as T
        import os
        path = ".h/data/corafull/graph.pt"
        if os.path.exists(path):
            graph = torch.load(path)
        else:
            
            dataset = CoraFull("data/corafull", transform=T.NormalizeFeatures())
            graph = dataset[0]
            
            # graph = split_ratio(graph)
            import ipdb; ipdb.set_trace()
            graph.valid_mask = graph.val_mask
            graph['valid_mask'] = graph['val_mask']
            graph['train_mask'] = graph['train_mask'][:, 0]
            graph['valid_mask'] = graph['valid_mask'][:, 0]
            torch.save(graph, path)
            # import ipdb; ipdb.set_trace()
        
        # import ipdb; ipdb.set_trace()
        print(graph)
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                    'valid': graph.valid_mask.nonzero().reshape(-1),
                    'test': graph.test_mask.nonzero().reshape(-1)}
    elif dataset_name.lower() in ["actor"]:
        from torch_geometric.datasets import Amazon, AttributedGraphDataset, CoraFull, Actor
        import torch_geometric.transforms as T
        import os
        path = "./data/actor/graph.pt"
        if os.path.exists(path):
            graph = torch.load(path)
        else:
            
            dataset = Actor("data/actor", transform=T.NormalizeFeatures())
            graph = dataset[0]
            
            # graph = split_ratio(graph)
            # import ipdb; ipdb.set_trace()
            graph.valid_mask = graph.val_mask
            graph['valid_mask'] = graph['val_mask']
            graph['train_mask'] = graph['train_mask'][:, 0]
            graph['valid_mask'] = graph['valid_mask'][:, 0]
            graph['test_mask'] = graph['test_mask'][:, 0]
            torch.save(graph, path)
            # import ipdb; ipdb.set_trace()
        
        
        print(graph)
        split_idx = {'train': graph.train_mask.nonzero().reshape(-1),
                    'valid': graph.valid_mask.nonzero().reshape(-1),
                    'test': graph.test_mask.nonzero().reshape(-1)}
        
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

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def split_ratio(data):
    size = data.num_nodes
    index = torch.arange(size)
    index = index[torch.randperm(index.size(0))]
    train_index = index[: int(size * 0.1)]
    val_index = index[int(size * 0.1): int(size * 0.3)]
    rest_index = index[int(size * 0.3):]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.valid_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

def random_coauthor_amazon_splits(data, num_classes, lcc_mask=None):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.valid_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data


# sythenic
def sanitize_transductive_task(data, k=1, resample=True, num_train_per_class=20,
                               num_val=500, num_test=1000, seed=321):
    """
    Greedily find a k-hop independent set of the graph nodes (i.e. all selected
    nodes are at least k+1 hops apart) and intersect it with all split masks
    of the @data i.e. data.{train/test/val}_mask.

    data: Single pytorch geometric graph data object
    k (int, optional): Size of the "buffer" between selected nodes
    resample (bool): Resample new dataset split masks
    num_train_per_class (int, optional): The number of training samples per class
    num_val (int, optional): The number of validation samples
    num_test (int, optional): The number of test samples
    seed (int, optional): Random seed
    :return: modified pytorch geometric graph data object
    """
    if k == 0:
        return data

    N = data.num_nodes
    A = [set() for _ in range(N)]
    for u, v in data.edge_index.t().tolist():
        A[u].add(v)
        A[v].add(u)

    for _ in range(k - 1):
        newA = [set() for _ in range(N)]
        for u in range(N):
            for v in A[u]:
                newA[u].update(A[v])
            newA[u].difference_update([u])  # remove self-loop
        A = newA

    rng = default_rng(seed=seed)
    nodes = [rng.choice(N)]  # list of initial "seed" nodes
    neighbors = set.union(*[A[v] for v in nodes])
    assert not set.intersection(neighbors, nodes), f"{nodes} is not a {k}-hop independent set of G"

    indep_nodes = list(nodes)
    available_nodes = set(range(N)).difference(neighbors.union(nodes))
    while available_nodes:
        node = rng.choice(list(available_nodes))
        indep_nodes.append(node)
        available_nodes.difference_update(list(A[node]) + [node])

    print(f"Found {k}-hop Independent Set of size {len(indep_nodes)}")
    indep_nodes = np.asarray(indep_nodes)

    train_nodes_before = data.train_mask.numpy().astype(int).sum()
    val_nodes_before = data.val_mask.numpy().astype(int).sum()
    test_nodes_before = data.test_mask.numpy().astype(int).sum()

    rm_mask = data.train_mask.new_empty(data.train_mask.size(0), dtype=torch.bool)
    rm_mask.fill_(True)
    rm_mask[indep_nodes] = False

    if resample:
        ys = data.y.clone().detach()
        ys[rm_mask] = -1  # don't pick masked-out nodes
        num_classes = ys.max().item() + 1

        if data.train_mask.ndimension() > 1:  # handling WikiCS dataset
            # supporting only a single data split
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.stopping_mask = data.stopping_mask[:, 0]

        data.train_mask.fill_(False)
        for c in range(num_classes):
            idx = (ys == c).nonzero(as_tuple=False).view(-1)
            idx = idx[rng.permutation(idx.size(0))[:num_train_per_class]]
            data.train_mask[idx] = True

        used = data.train_mask.clone().detach()
        used[rm_mask] = True
        remaining = (~used).nonzero(as_tuple=False).view(-1)
        remaining = remaining[rng.permutation(remaining.size(0))]
        num_remaining = remaining.size(0)

        num_needed = num_val + num_test + (num_val if hasattr(data, 'stopping_mask') else 0)
        print(f"> remaining: {num_remaining}, needed: {num_needed}")
        if num_needed > num_remaining:
            if hasattr(data, 'stopping_mask'):
                num_val = int(num_remaining * 0.25)
                num_test = int(num_remaining * 0.5)
            else:
                num_val = int(num_remaining * 0.333)
                num_test = int(num_remaining * 0.666)
            print(f"> new num_val {num_val}, num_test: {num_test}")

        data.val_mask.fill_(False)
        data.val_mask[remaining[:num_val]] = True
        num_prev = num_val

        if hasattr(data, 'stopping_mask'):
            stop_nodes_before = data.stopping_mask.numpy().astype(int).sum()
            data.stopping_mask.fill_(False)
            data.stopping_mask[remaining[num_prev:num_prev + num_val]] = True
            num_prev += num_val

        data.test_mask.fill_(False)
        data.test_mask[remaining[num_prev:num_prev + num_test]] = True

    else:
        data.train_mask[rm_mask] = False
        data.val_mask[rm_mask] = False
        data.test_mask[rm_mask] = False
        if hasattr(data, 'stopping_mask'):
            stop_nodes_before = data.stopping_mask.numpy().astype(int).sum()
            data.stopping_mask[rm_mask] = False

    train_nodes_after = data.train_mask.numpy().astype(int).sum()
    val_nodes_after = data.val_mask.numpy().astype(int).sum()
    test_nodes_after = data.test_mask.numpy().astype(int).sum()

    print(f">> Sanitizing... found Independent Set of size {len(indep_nodes)}")
    print(f"     train_nodes: before={train_nodes_before}, after={train_nodes_after}")
    print(f"     val_nodes:  before={val_nodes_before}, after={val_nodes_after}")
    if hasattr(data, 'stopping_mask'):
        stop_nodes_after = data.stopping_mask.numpy().astype(int).sum()
        print(f"     stop_nodes:  before={stop_nodes_before}, after={stop_nodes_after}")
    print(f"     test_nodes: before={test_nodes_before}, after={test_nodes_after}")
    print("     all y: ", np.unique(data.y.detach().numpy(), return_counts=True))

    return data


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




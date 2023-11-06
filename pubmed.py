from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit, Reddit2
import numpy as np
import torch

from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_geometric.datasets import Planetoid, Flickr, PPI
import ipdb

from load import GraphPreprocess

# Pubmed
# dataset = Planetoid('dataset', 'Pubmed', transform=GraphPreprocess(True, True))
# data = dataset[0]
# ipdb.set_trace()

# dataset = Flickr('./dataset/flickr', transform=GraphPreprocess(True, True))
# ipdb.set_trace()
# data = dataset[0]
# ipdb.set_trace()

dataset = Reddit('./dataset/reddit', pre_transform=GraphPreprocess(True, True))
ipdb.set_trace()
data = dataset[0]
ipdb.set_trace()


# dataset = PPI('./dataset/ppi', transform=GraphPreprocess(True, True))
# ipdb.set_trace()
# data = dataset[0]
# ipdb.set_trace()


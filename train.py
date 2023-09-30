import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
from model import SAGE, GCN

from utils import load_data, GraphPreprocess
from loader import VirtualClusterData


device = f'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(model, data, optimizer):
    model.train()

    optimizer.zero_grad()
    # import ipdb; ipdb.set_trace()
    out = model(data.x.to(device), data.edge_index.to(device))[data.train_mask]
    # import ipdb; ipdb.set_trace()
    loss = F.nll_loss(out, data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x.to(device), data.edge_index.to(device))
    y_pred = out.argmax(dim=-1, keepdims=True)
    # import ipdb; ipdb.set_trace
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']].unsqueeze(1),
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']].unsqueeze(1),
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']].unsqueeze(1),
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--use_virtual', action='store_true')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # 1. load data
    graph, split_idx = load_data("arxiv", small_trainingset=1.0, pretransform=GraphPreprocess(True, True))
    
    if args.use_virtual:
        graph = VirtualClusterData(graph, num_parts=100).data
    # import ipdb; ipdb.set_trace()
    # 2. dataloader

    # 3. model
    if args.use_sage:
        model = SAGE(graph.num_features, args.hidden_channels,
                     int(graph.y.max().item()) + 1, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(graph.num_features, args.hidden_channels,
                    int(graph.y.max().item()) + 1, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, graph, optimizer)
            result = test(model, graph, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
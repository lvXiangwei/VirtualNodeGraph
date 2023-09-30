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
from torch_geometric.loader import NeighborLoader
import wandb

from config import hyperparameter_defaults, sweep_config
from utils import ObjDict

device = f'cuda:{hyperparameter_defaults["device"]}' if torch.cuda.is_available() else 'cpu'

def train(model, data, optimizer, split_idx=None):
    model.train()

    optimizer.zero_grad()
    # import ipdb; ipdb.set_trace()
    out = model(data.x.to(device), data.edge_index.to(device))[data.train_mask]
    # import ipdb; ipdb.set_trace()
    loss = F.nll_loss(out, data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def batch_train(model, train_loader, optimizer, split_idx):
    # pbar = tqdm(total=split_idx['train'].size(0))
    # pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x.to(device), batch.edge_index.to(device))[:batch.batch_size]
        y = batch.y[:batch.batch_size].squeeze().to(device)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        # total_correct += int(out.argmax(dim=-1).eq(y).sum())
        # pbar.update(batch.batch_size)
    loss = total_loss / len(train_loader)
    # approx_acc = total_correct / split_idx['train'].size(0)
    return loss


@torch.no_grad()
def test(model, data, split_idx, evaluator, inference_mode='full', subgraph_loader = None):
    model.eval()
    if inference_mode == "full":
        out = model(data.x.to(device), data.edge_index.to(device))
    else:
        out = model.inference(data.x.to(device), device, subgraph_loader)

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


def main(config=hyperparameter_defaults):
    # with wandb.init(config=hyperparameter_defaults):
        # config = wandb.config
        # config.update()

    # 0. load config
    config = ObjDict(config)
    print(config)
    # import ipdb; ipdb.set_trace()
    
    # 1. load data
    graph, split_idx = load_data("arxiv", small_trainingset=config.small_trainingset, pretransform=GraphPreprocess(True, True))
    if config.use_virtual:
        graph = VirtualClusterData(graph, num_parts=config.num_parts).data
    
    # 2. dataloader
    # train dataloader 
    train_loader = NeighborLoader(
        graph, 
        input_nodes=graph.train_mask,
        num_neighbors=[config.neighbor1, config.neighbor2, config.neighbor3],
        batch_size = config.batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
    )

    # test dataloader 
    subgraph_loader = NeighborLoader(
        graph, 
        input_nodes=None,
        num_neighbors=[-1],
        batch_size = 4096,
        num_workers=1,
        persistent_workers=True,
    )
    
    # 3. model
    if config.model_name == "sage":
        model = SAGE(graph.num_features, config.hidden_channels,
                    int(graph.y.max().item()) + 1, config.num_layers,
                    config.dropout).to(device)
    elif config.model_name == "gcn":
        model = GCN(graph.num_features, config.hidden_channels,
                    int(graph.y.max().item()) + 1, config.num_layers,
                    config.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(config.runs, config)

    for run in range(config.runs):
        if config.wandb:
            wandb.init(
                project="virtualnodegraph", 
                name = f"{config.experiment_name}_{run}",
                config = hyperparameter_defaults,
            )
        config = wandb.config

        model.reset_parameters()
        if config.wandb:
            wandb.watch(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        for epoch in range(1, 1 + config.epochs):
            # loss = train(model, graph, optimizer)
            loss = batch_train(model, train_loader, optimizer, split_idx)
            
            if epoch % config.log_steps == 0:
                result = test(model, graph, split_idx, evaluator, config.inference_mode, subgraph_loader)
                logger.add_result(run, result)
                train_acc, valid_acc, test_acc = result
                # import ipdb; ipdb.set_trace()
                if config.wandb:
                    wandb.log({
                        "train/loss": loss,
                        "train/acc": train_acc,
                        "val/acc": valid_acc,
                        "test/acc": test_acc
                        },
                        step = epoch  
                    )
                print(f'Run: {0 + run:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')

        run_final_acc = logger.print_statistics(run)

        if config.wandb:
            wandb.summary["final_acc"] = run_final_acc
    final_acc, std = logger.print_statistics()


if __name__ == "__main__":
    main()
    # sweep_id = wandb.sweep(sweep_config, project="arxiv-neighborsampling")
    # wandb.agent(sweep_id, main)

    
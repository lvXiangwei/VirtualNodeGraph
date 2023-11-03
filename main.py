import torch
import torch.nn.functional as F
import wandb
import argparse

from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, JumpingKnowledge
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
from utils import count_parameters, seed_everything, load_config
from load import load_data, GraphPreprocess

from model3 import VNGNN

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    out = model(data)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, evaluator):
    model.eval()
    # out = model(data.x.to(device), data.edge_index.to(device))
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdims=True)
    
    # loss
    train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
    valid_loss = F.nll_loss(out[data.valid_mask], data.y[data.valid_mask].to(device))
    test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask].to(device))

    # acc
    train_acc = evaluator.eval({
        'y_true': data.y[data.train_mask].unsqueeze(1),
        'y_pred': y_pred[data.train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data.valid_mask].unsqueeze(1),
        'y_pred': y_pred[data.valid_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_mask].unsqueeze(1),
        'y_pred': y_pred[data.test_mask],
    })['acc']

    return (train_acc, valid_acc, test_acc), (train_loss, valid_loss, test_loss)


def main():
    parser = argparse.ArgumentParser(description='VirutalNodeGraph (GNN)')
    parser.add_argument('--dataset', type=str, default="arxiv") # reddit
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--model',type=str, default="gcn")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--experiment_name', type=str)
    
    parser.add_argument('--JK', action='store_true')
    parser.add_argument('--mode', type=str, default="max")
    
    parser.add_argument('--use_virtual', action='store_true')
    parser.add_argument("--cluster_method", type=str, default="metis")
    parser.add_argument('--num_parts', type=int, default=128)
    parser.add_argument('--sparse_ratio', type=float, default=1.0)

    # parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg', type=str, default=None) # for reproduction
    args = parser.parse_args()
    
    if args.cfg:
        # load config file
        cfg = load_config(args.cfg)
        # update argparse
        for key, value in cfg.items():
            setattr(args, key, value)
    
    print("Arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")

    
    global device
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # seed_everything(args.seed)
    # 1. load data
    graph, _ = load_data(args.dataset, small_trainingset=1.0, pretransform=GraphPreprocess(True, True))
    # import ipdb; ipdb.set_trace()
    # 2. model 
    #########################################################
    model = VNGNN(graph.num_features, 
                  args.hidden_channels, 
                  int(graph.y.max().item()) + 1,
                  args.num_layers, 
                  args.dropout, 
                  graph.num_nodes, 
                  graph.edge_index, 
                  dataset=args.dataset,
                  model=args.model,
                  clutser_method=args.cluster_method, 
                  num_clusters=args.num_parts, 
                  JK=args.JK, 
                  use_virtual=args.use_virtual,
                  mode=args.mode, 
                  sparse_ratio=args.sparse_ratio).to(device)
    #########################################################
    print("Parameters:")
    count_parameters(model)
    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)
    print("\nStart Training...")
    for run in range(args.runs):
        model.reset_parameters()
        # import ipdb; ipdb.set_trace()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if run == 0 and args.wandb:
            wandb.init(
                project = "obgn-arxiv",
                name = f"{args.experiment_name}",
            )
            wandb.config.update(args)
            
        for epoch in range(1, 1 + args.epochs):
            train(model, graph, optimizer)
            if epoch % args.log_steps == 0:
                accs, losses = test(model, graph, evaluator)
                logger.add_result(run, accs)
                (train_acc, valid_acc, test_acc), (train_loss, valid_loss, test_loss) = accs, losses
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Valid Loss: {valid_loss: .4f}, '
                      f'Test Loss: {test_loss: .4f}, ',
                      f'Train Acc: {100 * train_acc:.2f}%, '
                      f'Valid Acc: {100 * valid_acc:.2f}%, '
                      f'Test Acc: {100 * test_acc:.2f}%')
                if run == 0 and args.wandb:
                    wandb.log({
                            "train/loss": train_loss,
                            "train/acc": train_acc,
                            "val/loss": valid_loss,
                            "val/acc": valid_acc,
                            "test/loss": test_loss,
                            "test/acc": test_acc
                        },
                        step = epoch  
                    )
        run_final_acc = logger.print_statistics(run)
        if run == 0 and args.wandb:
            wandb.summary["final_acc"] = run_final_acc
    logger.print_statistics()

if __name__ == "__main__":
    main()


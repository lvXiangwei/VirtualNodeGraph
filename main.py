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
from torch_geometric.loader import NeighborLoader

from model import VNGNN

def bacth_train(model, train_loader, optimizer, criterion=F.nll_loss):
    model.train()
    total_loss = total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        batch.x = batch.x.to(device)
        if batch.edge_index is None:
            batch.edge_index = batch.adj_t
        batch.edge_index = batch.edge_index.to(device)
        out = model(batch)[batch.train_mask]
        if criterion == F.nll_loss:
            out = out.log_softmax(dim=-1)
        else:
            batch.y = batch.y.to(torch.float)
        loss = criterion(out, batch.y[batch.train_mask].to(device))
        loss.backward()
        optimizer.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes
    return total_loss / total_nodes

def train(model, data, optimizer, criterion=F.nll_loss):
    model.train()
    optimizer.zero_grad()
    data.x = data.x.to(device)
    if data.edge_index is None:
        data.edge_index = data.adj_t
    data.edge_index = data.edge_index.to(device)
    out = model(data)[data.train_mask]
    # import ipdb; ipdb.set_trace()
    if criterion == F.nll_loss:
        out = out.log_softmax(dim=-1)
    else:
        data.y = data.y.to(torch.float)
    loss = criterion(out, data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, evaluator, criterion=F.nll_loss):
    model.eval()
    # out = model(data.x.to(device), data.edge_index.to(device))
    data.x = data.x.to(device)
    if data.edge_index is None:
        data.edge_index = data.adj_t
    data.edge_index = data.edge_index.to(device)
    out = model(data, batch_train=False)

    if criterion == F.nll_loss:
        y_pred = out.argmax(dim=-1, keepdims=True)
        out = out.log_softmax(dim=-1)
    else:
        y_pred = out
        data.y = data.y.to(torch.float)

        
    train_loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
    valid_loss = criterion(out[data.valid_mask], data.y[data.valid_mask].to(device))
    test_loss = criterion(out[data.test_mask], data.y[data.test_mask].to(device))

    # import ipdb; ipdb.set_trace()
    if criterion  == F.nll_loss:
        train_metric = evaluator.eval({
            'y_true': data.y[data.train_mask].unsqueeze(1),
            'y_pred': y_pred[data.train_mask],
        })
        valid_metric = evaluator.eval({
            'y_true': data.y[data.valid_mask].unsqueeze(1),
            'y_pred': y_pred[data.valid_mask],
        })
        test_metric = evaluator.eval({
            'y_true': data.y[data.test_mask].unsqueeze(1),
            'y_pred': y_pred[data.test_mask],
        })
    else:
        train_metric = evaluator.eval({
            'y_true': data.y[data.train_mask],
            'y_pred': y_pred[data.train_mask],
        })
        valid_metric = evaluator.eval({
            'y_true': data.y[data.valid_mask],
            'y_pred': y_pred[data.valid_mask],
        })
        test_metric = evaluator.eval({
            'y_true': data.y[data.test_mask],
            'y_pred': y_pred[data.test_mask],
        })

    return (train_metric, valid_metric, test_metric), (train_loss, valid_loss, test_loss)


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
    parser.add_argument('--use_bn', type=int, default=1)
    parser.add_argument('--mlp_share', action='store_true')
    # parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optim', type=str, default="adam")
    
    parser.add_argument('--cfg', type=str, default=None) # for reproduction

    parser.add_argument('--sample', type=str, default=None) # 

    # for model 3
    parser.add_argument('--attn_dropout', type=float, default=-1) # 
    parser.add_argument('--merge_dropout', type=float, default=-1) # 
    parser.add_argument("--vntran_act", type=str, default="tanh")

    # for visualization
    parser.add_argument("--save_emb", action='store_true')
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
    if args.sample in ["neighbor"]:
        train_loader = NeighborLoader(
            graph, 
            input_nodes=graph.train_mask,
            num_neighbors=[6, 5, 5],
            batch_size = 1024,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
        )

        # test_loader = NeighborLoader(
        #     graph, 
        #     input_nodes=None,
        #     num_neighbors=[-1],
        #     batch_size = 4096,
        #     num_workers=1,
        #     persistent_workers=True,
        # )
    # 2. model 
    #########################################################
    # import ipdb; ipdb.set_trace()
    model = VNGNN(graph.num_features, 
                  args.hidden_channels, 
                  int(graph.y.max().item()) + 1 if graph.y.dim() == 1 else graph.y.shape[1],
                  args.num_layers, 
                  args.dropout, 
                  graph.num_nodes, 
                  graph.edge_index, 
                  dataset=args.dataset,
                  model=args.model,
                  cluster_method=args.cluster_method, 
                  num_clusters=args.num_parts, 
                  JK=args.JK, 
                  use_virtual=args.use_virtual,
                  mode=args.mode, 
                  sparse_ratio=args.sparse_ratio,
                  use_bn = args.use_bn,
                  mlp_share=args.mlp_share, 
                  args=args).to(device)
    #########################################################
    print("Parameters:")
    count_parameters(model)
    if args.dataset == "proteins":
        print("dataset name: ", args.dataset)
        evaluator = Evaluator(name='ogbn-proteins')
        metric = "rocauc"
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        evaluator = Evaluator(name='ogbn-arxiv')
        criterion = F.nll_loss
        metric = "acc"
    logger = Logger(args.runs, args)
    print("\nStart Training...")
    for run in range(args.runs):
        model.reset_parameters()
        # import ipdb; ipdb.set_trace()
        if args.optim == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if run == 0 and args.wandb:
            wandb.init(
                project = f"{args.dataset}",
                name = f"{args.experiment_name}",
            )
            wandb.config.update(args)
            
        for epoch in range(1, 1 + args.epochs):
            # import ipdb; ipdb.set_trace()
            best_valid_metric  = 0.0
            if args.sample == "neighbor":
                bacth_train(model, train_loader, optimizer, criterion)
            else:
                train(model, graph, optimizer, criterion)
            if epoch % args.log_steps == 0:
                metrics, losses = test(model, graph, evaluator, criterion)
                
                (train_metric, valid_metric, test_metric), (train_loss, valid_loss, test_loss) = metrics, losses
                if args.save_emb and valid_metric[metric] > best_valid_metric:
                    best_valid_metric = valid_metric[metric]
                    emb_data = {'emb': model(graph, save_emb=True), 
                                'test_mask': graph.test_mask,
                                'true_label': graph.y}
                    # import ipdb; ipdb.set_trace()
                    if args.save_emb:
                        emb_data["best_epoch"] = epoch
                        emb_data["test_result"] = test_metric[metric]
                        save_name = "emb/" + cfg['dataset'] + '_' + str(cfg['num_parts']) + ".pth"
                        torch.save(emb_data, save_name)
                
                logger.add_result(run, [val[metric] for val in metrics])
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Valid Loss: {valid_loss: .4f}, '
                      f'Test Loss: {test_loss: .4f}, ',
                      f'Train {metric.title()}: {train_metric[metric]:.4f}, '
                      f'Valid {metric.title()}: {valid_metric[metric]:.4f}, '
                      f'Test {metric.title()}: {test_metric[metric]:.4f}')
                if run == 0 and args.wandb:
                    wandb.log({
                            "train/loss": train_loss,
                            f"train/{metric.title()}": train_metric,
                            "val/loss": valid_loss,
                            f"val/{metric.title()}": valid_metric,
                            "test/loss": test_loss,
                            f"test/{metric.title()}": test_metric
                        },
                        step = epoch  
                    )
        run_final_metric = logger.print_statistics(run)
        if run == 0 and args.wandb:
            wandb.summary[f"final_{metric}"] = run_final_metric
    logger.print_statistics()

if __name__ == "__main__":
    main()


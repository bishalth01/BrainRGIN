import os
import sys
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from torch_geometric.nn import TopKPooling
from scipy.stats import pearsonr
import dgl
import argparse
from pytorchtools import EarlyStopping
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, explained_variance_score, 
    classification_report, confusion_matrix, r2_score
)
import wandb
from net.rgin_garo_model import CustomNetworkWithGARO


#Definining utility functions

def dgl_to_pyg(dgl_graph):
    """
    Convert a DGL graph object to a PyG data object.
    """
    edge_index = torch.stack(dgl_graph.edges(), dim=0)  # Get edge index in PyG format
    x = dgl_graph.ndata['x']  # Node features
    edge_attr = dgl_graph.edata.get('edge_attr', None)  # Optional: Include edge weights

    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data

def compute_metrics(outputs):
    all_preds = torch.cat([output for output, _ in outputs], dim=0).cpu().numpy().astype(np.float64)
    all_preds = np.nan_to_num(all_preds)  # Replace NaNs with 0

    all_labels = torch.cat([label for _, label in outputs], dim=0).cpu().numpy()
    all_labels = np.nan_to_num(all_labels)  # Replace NaNs with 0

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    correlation, _ = pearsonr(all_labels.flatten(), all_preds.flatten())
    r2 = r2_score(all_labels.flatten(), all_preds.flatten())

    return {'r': correlation, 'mse': mse, 'mae': mae, 'r2': r2}


#Defining loss functions

def topk_loss(s, ratio):
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values
    return -torch.log(s[:, -int(s.size(1) * ratio):] + EPS).mean() - torch.log(1 - s[:, :int(s.size(1) * ratio)] + EPS).mean()

def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0], s.shape[0])
    D = torch.eye(s.shape[0]) * torch.sum(W, dim=1)
    L = D - W
    L = L.to(device)
    return torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])


def braingnn_loss(opt, output, allpools, scores, loss_c, y):
    scores_list, s, pool_weights, loss_pools, loss_tpks = [], [], [], [], []

    for i in range(len(scores)):
        scores_list.append(torch.sigmoid(scores[i]).view(output.size(0), -1).view(-1).detach().cpu().numpy())
        s.append(torch.sigmoid(scores[i]).view(output.size(0), -1))

        module = allpools[i]
        module_params = [param for name, param in module.named_parameters() if param.requires_grad]
        pool_weights.extend(module_params)
        
        loss_pools.append((torch.norm(module_params[0], p=2) - 1) ** 2)  
        loss_tpks.append(topk_loss(s[i], opt.ratio))

    loss = opt.lamb0 * loss_c + opt.lamb1 * loss_pools[0] + opt.lamb2 * loss_pools[1] \
            + opt.lamb3 * loss_tpks[0] + opt.lamb4 * loss_tpks[1]

    return loss

#Dataset setup

class GraphDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for loading graph data.
    """
    def __init__(self, graphs, labels, indices):
        self.graphs = [graphs[i] for i in indices]
        self.labels = labels[indices]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]



def train_and_evaluate_model(opt, model, train_loader, val_loader, test_loader, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, verbose=True)
    early_stopping = EarlyStopping(patience=24, verbose=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    train_losses, val_losses = [], []
    train_correlations, val_correlations = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_outputs, all_labels = [], []

        for batch, labels in train_loader:
            labels = labels.to(device, non_blocking=True)
            batch_pos = batch.ndata['pos'].to(device, non_blocking=True)
            pyg_graph = dgl_to_pyg(batch)
            node_features = pyg_graph.x.to(device, non_blocking=True)
            edge_index = pyg_graph.edge_index.to(device, non_blocking=True)
            edge_weights = pyg_graph.edge_attr.to(device, non_blocking=True)

            batch_nodes = torch.arange(int(batch.number_of_nodes() / opt.nroi), device=device).repeat_interleave(opt.nroi)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Mixed precision context
                outputs, allpools, scores = model(node_features, edge_index, batch_nodes, edge_weights, batch_pos)
                loss_c = criterion(outputs.float(), labels.float())
                loss = braingnn_loss(opt, outputs, allpools, scores, loss_c, labels.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        train_metrics = compute_metrics([(all_outputs.cpu(), all_labels.cpu())])

        val_metrics = evaluate_model(opt, model, val_loader, device)
        val_loss = val_metrics['mse']

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
            
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        print(
            f'Epoch: {epoch:03d}, Loss: {avg_train_loss:.7f} / {val_loss:.7f}, '
            f'R2: {train_metrics["r2"]:.4f} / {val_metrics["r2"]:.4f}, R: {train_metrics["r"]:.4f} / {val_metrics["r"]:.4f}, '
            f'MSE: {train_metrics["mse"]:.4f} / {val_metrics["mse"]:.4f}, MAE: {train_metrics["mae"]:.4f} / {val_metrics["mae"]:.4f}'
        )
    
        wandb.log({
            'train_loss': train_metrics['mse'], 'val_loss': val_metrics['mse'],
            'train_r2': train_metrics['r2'], 'val_r2': val_metrics['r2'],
            'train_r': train_metrics['r'], 'val_r': val_metrics['r'],
            'train_mse': train_metrics['mse'], 'val_mse': val_metrics['mse'],
            'train_mae': train_metrics['mae'], 'val_mae': val_metrics['mae']
        })

        if val_loss < best_loss and epoch > 5:
            print("Saving best model")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if opt.save_model:
                torch.save(model.state_dict(), os.path.join(opt.save_path, "best_model_cryst_abcd_garo.pth"))

    model.load_state_dict(best_model_wts)
    test_metrics = evaluate_model(opt, model, test_loader, device)
    print(f'Test results - R: {test_metrics["r"]:.4f}, MSE: {test_metrics["mse"]:.4f}, MAE: {test_metrics["mae"]:.4f}, R2: {test_metrics["r2"]:.4f}')


def evaluate_model(opt, model, loader, device):
    model.eval()
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for batch, labels in loader:
            labels = labels.to(device)
            batch_pos = batch.ndata['pos'].to(device, non_blocking=True)
            pyg_graph = dgl_to_pyg(batch)
            node_features = pyg_graph.x.to(device, non_blocking=True)
            edge_index = pyg_graph.edge_index.to(device, non_blocking=True)
            edge_weights = pyg_graph.edge_attr.to(device, non_blocking=True)

            batch_nodes = torch.arange(int(batch.number_of_nodes() / opt.nroi), device=device).repeat_interleave(opt.nroi)
            with torch.cuda.amp.autocast():  # Mixed precision context
                outputs, _, _ = model(node_features, edge_index, batch_nodes, edge_weights, batch_pos)

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return compute_metrics([(all_outputs.cpu(), all_labels.cpu())])


if __name__ == "__main__":

    def set_random_seeds(seed=1111):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    EPS = 1e-10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ['WANDB_DISABLE_CODE'] = 'false'
    wandb.init(project='uncategorized', save_code=True, config="wandb_sweeps/example_config.yaml")
    config = wandb.config
    set_random_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='data/Output', help='root directory of the dataset')
    parser.add_argument('--stepsize', type=int, default=30, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='scheduler shrinking rate')
    parser.add_argument('--indim', type=int, default=53, help='feature dim')
    parser.add_argument('--nroi', type=int, default=53, help='num of ROIs')
    parser.add_argument('--nclass', type=int, default=1, help='num of classes')
    parser.add_argument('--fold', type=int, default=0, help='training which fold')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='./model_outputs/', help='path to save model')
    # Arguments from WANDB Sweeps
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--epoch', type=int, default=config.epoch, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=config.n_epochs, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=config.batchSize, help='size of the batches')
    parser.add_argument('--weightdecay', type=float, default=config.weightdecay, help='regularization')
    parser.add_argument('--lamb0', type=float, default=config.lamb0, help='classification loss weight')
    parser.add_argument('--lamb1', type=float, default=config.lamb1, help='s1 unit regularization')
    parser.add_argument('--lamb2', type=float, default=config.lamb2, help='s2 unit regularization')
    parser.add_argument('--lamb3', type=float, default=config.lamb3, help='s1 entropy regularization')
    parser.add_argument('--lamb4', type=float, default=config.lamb4, help='s2 entropy regularization')
    parser.add_argument('--lamb5', type=float, default=config.lamb5, help='s1 consistence regularization')
    parser.add_argument('--reg', type=float, default=0.1, help='GMT reg')
    parser.add_argument('--layer', type=int, default=config.layer, help='number of GNN layers')
    parser.add_argument('--ratio', type=float, default=config.ratio, help='pooling ratio')
    parser.add_argument('--optim', type=str, default=config.optim, help='optimization method: SGD, Adam')
    parser.add_argument('--n_layers', type=str, default=config.n_layers, help='Dimensions of hidden layers')
    parser.add_argument('--n_fc_layers', type=str, default=config.n_fc_layers, help='Dimensions of fully connected layers')
    parser.add_argument('--n_clustered_communities', type=int, default=config.n_clustered_communities, help='Number of clustered communities')
    parser.add_argument('--early_stop_steps', type=int, default=config.early_stop_steps, help='Early Stopping Steps')

    opt = parser.parse_args()

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    # opt.n_layers = [int(n) for n in str(opt.n_layers[0]).split(',')]
    # opt.n_fc_layers = [int(n) for n in str(opt.n_fc_layers[0]).split(',')]

    opt.n_layers = [int(float(numeric_string)) for numeric_string in opt.n_layers.split(',')]
    opt.n_fc_layers = [int(float(numeric_string)) for numeric_string in opt.n_fc_layers.split(',')]

    # Load datasets
    load_graphs = torch.load("abcd_graphs_crystallized.pt")
    load_labels = torch.load("/abcd_graphs_crystallized_labels.pt")

    # Split datasets
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)

    num_graphs = len(load_graphs)
    indices = np.random.permutation(num_graphs)
    train_size, val_size = int(train_ratio * num_graphs), int(val_ratio * num_graphs)
    test_size = num_graphs - train_size - val_size

    tr_index, val_index, te_index = indices[:train_size], indices[train_size:train_size + val_size], indices[train_size + val_size:]

    # Create datasets and dataloaders
    train_dataset = GraphDataset(load_graphs, load_labels, tr_index)
    val_dataset = GraphDataset(load_graphs, load_labels, val_index)
    test_dataset = GraphDataset(load_graphs, load_labels, te_index)

    train_dataloader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, drop_last=False,  pin_memory=True)
    val_dataloader = dgl.dataloading.GraphDataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False, drop_last=False,  pin_memory=True)
    test_dataloader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, drop_last=False,  pin_memory=True)

    model = CustomNetworkWithGARO(indim= opt.indim,ratio = opt.ratio,nclass = opt.nclass, n_hidden_layers=opt.n_layers, n_fc_layers=opt.n_fc_layers,k = opt.n_clustered_communities,R = opt.nroi)
    train_and_evaluate_model(opt, model, train_dataloader, val_dataloader, test_dataloader, num_epochs=500)

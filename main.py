import sys
import os
import csv
import copy

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader

from config import CONFIG, ClassificationType
from utils import set_seed, k_fold
from diffusion_search import get_diffusion_params
from pA_GAT import train_graph, test_graph, train_node, test_node, WeightedGATGraphNet
from dataset_loader import prepare_node_dataset, prepare_graph_dataset, NODE_DATASET_FAMILY
from baselines import DiffPool, GAT, GCN, GCN2, GIN, GraphSAGE, GPRGNN, H2GCN, GREAD, GraphGPS 
from itertools import product

def get_model(model_name, num_features, num_classes, max_nodes, hidden_dim, num_layers, dropout_rate, regression=False, task_level='graph'):
    """Instantiates a fresh model for each fold."""
    if model_name == "PAGAT":
        return WeightedGATGraphNet(
            in_dim=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            heads=4,
            regression=regression,
            task_level=task_level
            )
    elif model_name == "GAT":
        return GAT(
                num_features=num_features,
                num_classes=num_classes,
                num_layers=num_layers,
                hidden=hidden_dim,
                heads=4,
                regression=regression,
                task_level=task_level
                )
    elif model_name == "GraphSAGE":
        return GraphSAGE(
                num_features=num_features,
                num_classes=num_classes,
                num_layers=num_layers,
                hidden=hidden_dim,
                regression=regression,
                task_level=task_level
                )
    elif model_name == "GIN":
        return GIN(
            num_features=num_features,
            num_classes=num_classes,
            num_layers=num_layers,
            hidden=hidden_dim,
            regression=regression
            )
    elif model_name == "GCN":
        return GCN(
            num_features=num_features,
            num_classes=num_classes,
            num_layers=num_layers,
            hidden=hidden_dim,
            regression=regression,
            task_level=task_level
            )
    elif model_name == "GCN2":
        return GCN2(
            num_features=num_features,
            num_classes=num_classes,
            num_layers=num_layers,
            hidden=hidden_dim,
            regression=regression,
            task_level=task_level
            )
    elif model_name == "DiffPool":
        return DiffPool(
            in_dim=num_features, num_classes=num_classes,
            num_layers=num_layers, hidden=hidden_dim,
            max_nodes=max_nodes, ratio=0.25,
            regression=regression
            )
    elif model_name == "GPRGNN":
        return GPRGNN(
                num_classes=num_classes,
                num_features=num_features,
                num_layers=num_layers,
                hidden=hidden_dim,
                regression=regression,
                task_level=task_level
                )
    elif model_name == "H2GCN":
        return H2GCN(
                num_classes=num_classes,
                num_features=num_features,
                num_layers=num_layers,
                hidden=hidden_dim,
                regression=regression,
                task_level=task_level
                )
    elif model_name == "GREAD":
        return GREAD(
                num_classes=num_classes,
                num_features=num_features,
                num_layers=num_layers,
                hidden=hidden_dim
                )
    elif model_name == "GraphGPS":
        return GraphGPS(
                num_classes=num_classes,
                num_features=num_features,
                num_layers=num_layers,
                hidden=hidden_dim
                )
    else:
        raise ValueError(f"Unknown model {model_name}")

def train_graph_diffpool(model, loader, optimizer, device, task_type='classification'):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None
        out = model(batch.x, batch.edge_index, batch=batch.batch, edge_weight=edge_weight)
        if task_type == 'classification':
            loss = F.nll_loss(out, batch.y)
        else:
            loss = F.mse_loss(out.squeeze(), batch.y.float())
        loss = loss + model.link_loss + model.ent_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ---------------- Graph classification/regression experiment ---------------
def run_experiment(model_name, dataset, splits, num_features, num_classes, device, max_nodes, hidden_dim, num_layers, dropout_rate, lr, weight_decay, regression=False):
    train_indices, val_indices, test_indices = splits
    all_test_acc = []
    task_type = 'regression' if regression else 'classification'

    fold_bar = tqdm(range(len(train_indices)), desc=f"{model_name} folds", leave=False)
    for fold_idx in fold_bar:
        set_seed(42 + fold_idx)

        train_dataset = [dataset[i] for i in train_indices[fold_idx]]
        val_dataset   = [dataset[i] for i in val_indices[fold_idx]]
        test_dataset  = [dataset[i] for i in test_indices[fold_idx]]

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

        model = get_model(
                model_name=model_name,
                num_features=num_features,
                num_classes=num_classes,
                max_nodes=max_nodes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                regression=regression
                ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if regression:
            best_val_score = float('inf')
        else:
            best_val_score = 0
        best_model_weights = None
        patience = 0
        early_stop = 100

        epoch_bar = tqdm(range(1, 3001), desc=f"{model_name} fold {fold_idx+1}/{len(train_indices)} epochs", leave=False)
        for epoch in epoch_bar:
            if model_name == "DiffPool":
                train_graph_diffpool(model, train_loader, optimizer, device, task_type=task_type)
            else:
                train_graph(model, train_loader, optimizer, device, task_type=task_type)

            val_acc = test_graph(model, val_loader, device, task_type=task_type)

            if val_acc > best_val_score and not regression:
                best_val_score = val_acc
                patience = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            elif regression and val_acc < best_val_score:
                best_val_score = val_acc
                patience = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                patience += 1

            epoch_bar.set_postfix(val=f"{val_acc:.4f}", best=f"{best_val_score:.4f}", patience=patience)

            if patience > early_stop:
                epoch_bar.set_description(f"{model_name} fold {fold_idx+1}/{len(train_indices)} early stop @ {epoch}")
                break
        epoch_bar.close()

        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)

        test_acc = test_graph(model, test_loader, device, task_type=task_type)
        fold_bar.set_postfix(test_acc=f"{test_acc:.4f}")
        all_test_acc.append(test_acc)

    mean_acc = np.mean(all_test_acc)
    std_acc = np.std(all_test_acc)

    return mean_acc, std_acc

# ---------------- Node classification experiment ---------------
def run_node_experiment(model_name, data, num_features, num_classes, device, hidden_dim, num_layers, dropout_rate, lr, weight_decay):
    set_seed(42)

    model = get_model(
            model_name=model_name,
            num_features=num_features,
            num_classes=num_classes,
            max_nodes=data.num_nodes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            task_level='node'
            ).to(device)
    data = data.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_score = 0
    best_model_weights = None
    patience = 0
    early_stop = 100

    epoch_bar = tqdm(range(1, 3001), desc=f"{model_name} epochs", leave=False)
    for epoch in epoch_bar:
        train_node(model, data, optimizer, device)
        val_acc = test_node(model, data, device, mask_name='val_mask')

        if val_acc > best_val_score:
            best_val_score = val_acc
            patience = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience += 1

        epoch_bar.set_postfix(val=f"{val_acc:.4f}", best=f"{best_val_score:.4f}", patience=patience)

        if patience > early_stop:
            epoch_bar.set_description(f"{model_name} early stop @ {epoch}")
            break
    epoch_bar.close()

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    test_acc = test_node(model, data, device, mask_name='test_mask')
    return test_acc

def run_node_task(input_dataset, hyperparams, device, results_file, pbar=None):
    if input_dataset not in NODE_DATASET_FAMILY:
        raise ValueError(f"Unknown node classification dataset {input_dataset}")

    diffusion_params = get_diffusion_params(input_dataset, task='node', device=device)

    dataset_unweighted, num_classes, num_features = prepare_node_dataset(
            root=f'data/Node/unweighted/{input_dataset}', name=input_dataset)
    dataset_weighted, _, _ = prepare_node_dataset(
            root=f'data/Node/weighted/{input_dataset}', name=input_dataset, diffusion_params=diffusion_params)

    for model_name in CONFIG.experiment.models:
        current_data = dataset_weighted if model_name == "PAGAT" else dataset_unweighted

        max_mean = -1
        saved_hidden_dim = saved_num_layers = saved_dropout_rate = saved_lr = saved_weight_decay = -1
        for hidden_dim, num_layers, dropout_rate, lr, weight_decay in hyperparams:
            if pbar is not None:
                pbar.set_description(f"{input_dataset} | {model_name} | h{hidden_dim} l{num_layers} d{dropout_rate} lr{lr} wd{weight_decay}")
            test_acc = run_node_experiment(model_name, current_data, num_features, num_classes, device, hidden_dim, num_layers, dropout_rate, lr, weight_decay)
            if test_acc > max_mean:
                max_mean = test_acc
                saved_hidden_dim = hidden_dim
                saved_num_layers = num_layers
                saved_dropout_rate = dropout_rate
                saved_lr = lr
                saved_weight_decay = weight_decay
            if pbar is not None:
                pbar.set_postfix(best=f"{max_mean:.4f}")
                pbar.update(1)

        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, input_dataset, max_mean, 0.0, saved_hidden_dim, saved_num_layers, saved_dropout_rate, saved_lr, saved_weight_decay])

def run_graph_task(input_dataset, hyperparams, device, results_file, pbar=None):
    diffusion_params = get_diffusion_params(input_dataset, task='graph', device=device)

    dataset_unweighted, num_classes, num_features, splits, generate_splits, regression = prepare_graph_dataset(input_dataset, diffusion_params=None)
    dataset_weighted, _, _, _, _, _ = prepare_graph_dataset(input_dataset, diffusion_params=diffusion_params)

    max_nodes_unweighted = max(data.num_nodes for data in dataset_unweighted)

    for model_name in CONFIG.experiment.models:
        current_dataset = dataset_weighted if model_name == "PAGAT" else dataset_unweighted

        current_splits = k_fold(current_dataset, folds=10, seed=42) if generate_splits else splits

        max_mean = float('inf') if regression else -1
        saved_std = -1
        saved_hidden_dim = saved_num_layers = saved_dropout_rate = saved_lr = saved_weight_decay = -1
        for hidden_dim, num_layers, dropout_rate, lr, weight_decay in hyperparams:
            if pbar is not None:
                pbar.set_description(f"{input_dataset} | {model_name} | h{hidden_dim} l{num_layers} d{dropout_rate} lr{lr} wd{weight_decay}")
            mean, std = run_experiment(model_name, current_dataset, current_splits, num_features, num_classes, device, max_nodes_unweighted, hidden_dim, num_layers, dropout_rate, lr, weight_decay, regression=regression)
            if (mean > max_mean and not regression) or (regression and mean < max_mean):
                max_mean = mean
                saved_std = std
                saved_hidden_dim = hidden_dim
                saved_num_layers = num_layers
                saved_dropout_rate = dropout_rate
                saved_lr = lr
                saved_weight_decay = weight_decay
            if pbar is not None:
                pbar.set_postfix(best=f"{max_mean:.4f}")
                pbar.update(1)

        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, input_dataset, max_mean, saved_std, saved_hidden_dim, saved_num_layers, saved_dropout_rate, saved_lr, saved_weight_decay])

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyperparams = list(product(
        CONFIG.hyperparameters.hidden_dim,
        CONFIG.hyperparameters.num_layers,
        CONFIG.hyperparameters.dropout_rate,
        CONFIG.hyperparameters.lr,
        CONFIG.hyperparameters.weight_decay,
    ))

    os.makedirs("./results", exist_ok=True)
    results_file = "./results/results.csv"

    datasets = CONFIG.experiment.datasets
    total_runs = len(datasets) * len(CONFIG.experiment.models) * len(hyperparams)

    with tqdm(total=total_runs, desc="Experiment", unit="run") as pbar:
        for input_dataset in datasets:
            if CONFIG.experiment.type == ClassificationType.NODE:
                run_node_task(input_dataset, hyperparams, device, results_file, pbar=pbar)
            else:
                run_graph_task(input_dataset, hyperparams, device, results_file, pbar=pbar)

    return 0

if __name__ == "__main__":
    sys.exit(main())

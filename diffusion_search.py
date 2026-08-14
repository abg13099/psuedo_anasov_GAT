import json
import os

import optuna
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import CONFIG
from dataset_loader import prepare_node_dataset, prepare_graph_dataset, NODE_DATASET_FAMILY
from pA_GAT import WeightedGATGraphNet, train_node, test_node, train_graph, test_graph
from utils import set_seed, k_fold

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _sample_diffusion_params(trial):
    space = CONFIG.diffusion_search.search_space
    alpha = trial.suggest_float('alpha', *space.alpha)
    base_anisotropy_c = trial.suggest_float('base_anisotropy_c', *space.base_anisotropy_c)
    beta = trial.suggest_float('beta', *space.beta, log=True)
    return dict(alpha=alpha, base_anisotropy_c=base_anisotropy_c, beta=beta)

def _build_proxy_model(num_features, num_classes, regression, task_level, device):
    cfg = CONFIG.diffusion_search
    model = WeightedGATGraphNet(
        in_dim=num_features,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_classes=num_classes,
        dropout_rate=cfg.dropout_rate,
        heads=4,
        regression=regression,
        task_level=task_level,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return model, optimizer

def _make_node_objective(dataset_name, device):
    def objective(trial):
        cfg = CONFIG.diffusion_search
        diffusion_params = _sample_diffusion_params(trial)
        data, num_classes, num_features = prepare_node_dataset(
                root=f'data/Node/weighted/{dataset_name}', name=dataset_name, diffusion_params=diffusion_params)

        set_seed(cfg.seed)
        model, optimizer = _build_proxy_model(num_features, num_classes, regression=False, task_level='node', device=device)
        data = data.to(device)

        best_val = 0.0
        patience = 0
        for _ in range(1, cfg.epochs + 1):
            train_node(model, data, optimizer, device)
            val_acc = test_node(model, data, device, mask_name='val_mask')
            if val_acc > best_val:
                best_val = val_acc
                patience = 0
            else:
                patience += 1
            if patience > cfg.patience:
                break
        return best_val
    return objective

def _make_graph_objective(dataset_name, device):
    regression = (dataset_name == 'ZINC')
    task_type = 'regression' if regression else 'classification'

    def objective(trial):
        cfg = CONFIG.diffusion_search
        diffusion_params = _sample_diffusion_params(trial)
        dataset, num_classes, num_features, splits, generate_splits, _ = prepare_graph_dataset(
                dataset_name, diffusion_params=diffusion_params)

        if generate_splits:
            splits = k_fold(dataset, folds=10, seed=cfg.seed)
        train_idx, val_idx, _ = splits
        # A single split is enough signal for ranking candidate diffusion
        # params; running full k-fold CV per trial would be far too slow.
        train_loader = DataLoader([dataset[i] for i in train_idx[0]], batch_size=32, shuffle=True)
        val_loader = DataLoader([dataset[i] for i in val_idx[0]], batch_size=32, shuffle=False)

        set_seed(cfg.seed)
        model, optimizer = _build_proxy_model(num_features, num_classes, regression=regression, task_level='graph', device=device)

        best_val = float('inf') if regression else 0.0
        patience = 0
        for _ in range(1, cfg.epochs + 1):
            train_graph(model, train_loader, optimizer, device, task_type=task_type)
            val_score = test_graph(model, val_loader, device, task_type=task_type)
            improved = (val_score < best_val) if regression else (val_score > best_val)
            if improved:
                best_val = val_score
                patience = 0
            else:
                patience += 1
            if patience > cfg.patience:
                break
        return best_val
    return objective

def _load_cache():
    path = CONFIG.diffusion_search.cache_path
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    path = CONFIG.diffusion_search.cache_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(cache, f, indent=2)

def get_diffusion_params(dataset_name, task, device=None):
    """Returns the best (alpha, base_anisotropy_c, beta) for `dataset_name`,
    running a short cached Optuna study the first time a dataset is requested.

    `task` is 'node' or 'graph'. Results are cached to
    CONFIG.diffusion_search.cache_path so repeat runs (e.g. across
    hyperparameter grid search in main.py) reuse the search instead of
    re-running it.
    """
    cache = _load_cache()
    key = f"{task}:{dataset_name}"
    if key in cache:
        cached = cache[key]
        return {k: cached[k] for k in ("alpha", "base_anisotropy_c", "beta")}

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if task == 'node':
        if dataset_name not in NODE_DATASET_FAMILY:
            raise ValueError(f"Unknown node classification dataset {dataset_name}")
        objective = _make_node_objective(dataset_name, device)
        direction = 'maximize'
    else:
        objective = _make_graph_objective(dataset_name, device)
        direction = 'minimize' if dataset_name == 'ZINC' else 'maximize'

    cfg = CONFIG.diffusion_search
    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    study = optuna.create_study(direction=direction, sampler=sampler)

    pbar = tqdm(total=cfg.n_trials, desc=f"diffusion search: {dataset_name}", unit="trial", leave=False)
    def _on_trial_done(study, trial):
        pbar.set_postfix(best=f"{study.best_value:.4f}")
        pbar.update(1)

    study.optimize(objective, n_trials=cfg.n_trials, callbacks=[_on_trial_done])
    pbar.close()

    best_params = study.best_params
    cache[key] = {**best_params, "value": study.best_value}
    _save_cache(cache)
    return best_params

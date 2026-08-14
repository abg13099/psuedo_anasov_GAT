import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def k_fold(dataset, folds=10, seed=42):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = [data.y.item() for data in dataset]

    # Get the 10 distinct fold buckets
    fold_indices = []
    for _, test_idx in skf.split(torch.zeros(len(dataset)), labels):
        fold_indices.append(test_idx)

    train_indices, val_indices, test_indices = [], [], []

    for i in range(folds):
        test_idx = fold_indices[i]
        val_idx = fold_indices[(i + 1) % folds]
        train_idx = np.concatenate([
            fold_indices[j] for j in range(folds) if j != i and j != (i + 1) % folds
        ])

        train_indices.append(torch.tensor(train_idx, dtype=torch.long))
        val_indices.append(torch.tensor(val_idx, dtype=torch.long))
        test_indices.append(torch.tensor(test_idx, dtype=torch.long))

    return train_indices, val_indices, test_indices

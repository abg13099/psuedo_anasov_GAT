import ml_collections as mlc

from enum import Enum, auto

class ClassificationType(Enum):
    NODE = auto()
    GRAPH = auto()

CONFIG = mlc.ConfigDict(
    {
        "experiment": {
                "type": ClassificationType.NODE,
                "datasets": ("Cora", "CiteSeer", "PubMed", "Texas", "Wisconsin", "Cornell", "CS", "Physics"),
                "models": ("GPRGNN", "H2GCN", "GCN2", "GREAD"),
        },
        # Diffusion edge weights (used by PAGAT) are tuned per-dataset by
        # diffusion_search.py rather than fixed globally: each dataset gets its
        # own short Optuna study over (alpha, base_anisotropy_c, beta), scored
        # by a small proxy PAGAT model, and the winning params are cached to
        # cache_path so the search only runs once per dataset.
        "diffusion_search": {
                "n_trials": 20,
                "epochs": 50,
                "patience": 15,
                "hidden_dim": 32,
                "num_layers": 2,
                "dropout_rate": 0.5,
                "lr": 1e-2,
                "weight_decay": 5e-4,
                "seed": 42,
                "cache_path": "cache/diffusion_params.json",
                "search_space": {
                        "alpha": (0.05, 1.0),
                        "base_anisotropy_c": (0.1, 5.0),
                        "beta": (0.001, 0.5),
                },
        },
        "hyperparameters": {
                "hidden_dim": (64,),
                "num_layers": (3, 4),
                "dropout_rate": (0, 0.1, 0.5),
                "lr": (1e-2, 5e-4, 1e-6),
                "weight_decay": (0, 1e-4),
        },
    }
)

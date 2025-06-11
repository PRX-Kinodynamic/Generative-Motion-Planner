from multiprocessing import cpu_count
from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product
import numpy as np
from genMoPlan.utils import watch, watch_dict

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

exp_args_to_watch = [
    ("history_length", "HILEN"),
    ("horizon_length", "HOLEN"),
    ("use_history_padding", "HIPAD"),
    ("use_horizon_padding", "HOPAD"),
    ("stride", "STRD"),
]

results_args_to_watch = [
    ("n_runs", "NRUN"),
    ("attractor_dist_threshold", "ADTH"),
    ("attractor_prob_threshold", "APTH"),
]

logbase = "experiments"

base = {
    "base": {
        "action_indices": [4, 5],
        "loss_type": "l2",
        "loss_weights": None,
        "loss_discount": 1,
        "clip_denoised": False,
        "observation_dim": 6,
        "has_local_query": False,
        "has_global_query": False,
        "history_length": 5,
        "horizon_length": 3,
        "stride": 30,

        #-------------------------------- dataset --------------------------------#
        "loader": "datasets.AcrobotDataset",
        "trajectory_normalizer": "LimitsNormalizer",
        "plan_normalizer": None,
        "normalizer_params": {
            "trajectory": {
                "mins": [-4 * np.pi, -4 * np.pi, -20.0, -20.0, -0.5, -6.0],
                "maxs": [4 * np.pi, 4 * np.pi, 20.0, 20.0, 0.5, 6.0],
            },
            "plan": None,
        },
        "plan_preprocess_fns": None,    
        "trajectory_preprocess_fns": [],
        "preprocess_kwargs": None,
        "use_history_padding": True,
        "use_horizon_padding": False,
        "use_plan": False,
        "train_dataset_size": None,
        "is_history_conditioned": True,
        "dataset_kwargs": {
            "use_original_data": False,
            "only_optimal": True,
            "velocity_limit": 20.0,
        },

        #---------------------------- serialization ----------------------------#
        "logbase": logbase,
        "exp_name": watch(exp_args_to_watch),

        #---------------------------- training ----------------------------#
        "num_epochs": 500,
        "min_num_batches_per_epoch": 1e4,
        "save_freq": 5, # epochs
        "log_freq": 1e3, # steps
        "batch_size": 256,
        "learning_rate": 1e-4,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.995,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": None,
        "num_workers": 10,

        #---------------------------- validation ----------------------------#
        "val_dataset_size": 8,
        "val_batch_size": 2048,
        "patience": 10,
        "early_stopping": False,

        "no_inference": True,
    },

    "flow_matching": {
        "method_name": "flow_matching",
        "method": "models.generative.FlowMatching",
        "manifold": None,
        "model": "models.temporal.TemporalUnet",
        "model_kwargs": {
            "base_hidden_dim": 64,
            "hidden_dim_mult": (1, 2, 4, 8),
            "conv_kernel_size": 5,
            "attention": False,
            "time_embed_dim": 32,
        },
        "method_kwargs": {
            "scheduler": "CondOTScheduler",
            "path": "AffineProbPath",
            "solver": "ODESolver",
            "n_fourier_features": 1,
        },
        "prefix": "flow_matching/",
        "min_delta": 1e-3,
        "validation_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
        },
    }
}
from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product
import numpy as np
from genMoPlan.utils import watch, handle_angle_wraparound, augment_unwrapped_state_data, watch_dict

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
    "roa_estimation": {
        "results_name": watch_dict(results_args_to_watch),
        "attractors": {
            (-2.1, 0): 0,
            (2.1, 0): 0,
            (0, 0): 1,
        },
        "invalid_label": -1,
        "n_runs": 100,
        "batch_size": int(1e6),
        "attractor_dist_threshold": 0.05,
        "attractor_prob_threshold": 0.98,
        "max_path_length": 502,
        "flow_matching": {
            "n_timesteps": 5,
            "integration_method": "euler",
        },
        "manifold": None,
    },

    "base": {
        "action_indices": None,
        "loss_type": "l2",
        "loss_weights": None,
        "loss_discount": 1,
        "clip_denoised": False,
        "observation_dim": 2,
        "has_query": False,

        #-------------------------------- dataset --------------------------------#
        "loader": "datasets.TrajectoryDataset",
        "trajectory_normalizer": "LimitsNormalizer",
        "plan_normalizer": None,
        "normalizer_params": {
            "trajectory": {
                "mins": [-2*np.pi, -2*np.pi],
                "maxs": [2*np.pi, 2*np.pi],
            },
            "plan": None,
        },
        "plan_preprocess_fns": None,    
        "trajectory_preprocess_fns": [
            handle_angle_wraparound,
            augment_unwrapped_state_data,
        ],
        "preprocess_kwargs": {
            "trajectory": {
                "angle_indices": [0],
            },
            "plan": None,
        },
        "use_history_padding": False,
        "use_horizon_padding": True,
        "use_plan": False,
        "train_dataset_size": None,
        "is_history_conditioned": True,

        #---------------------------- serialization ----------------------------#
        "logbase": logbase,
        "exp_name": watch(exp_args_to_watch),

        #---------------------------- training ----------------------------#
        "num_epochs": 100,
        "min_num_batches_per_epoch": 1e4,
        "save_freq": 20, # epochs
        "log_freq": 1e3, # steps
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": None,

        #---------------------------- validation ----------------------------#
        "val_dataset_size": 100,
        "val_num_batches": 10,
        "patience": 10,
        "early_stopping": True,
    },

    "diffusion": {
        "method_name": "diffusion",
        "model": "models.temporal.TemporalUnet",
        "method": "models.generative.Diffusion",
        "horizon_length": 31,
        "history_length": 1,
        "stride": 1,
        
        "model_kwargs": {
            "base_hidden_dim": 32,
            "hidden_dim_mult": (1, 2, 4, 8),
            "conv_kernel_size": 5,
            "attention": False,
        },
        "method_kwargs": {
            "predict_epsilon": False,
            "n_timesteps": 20,
        },
        "prefix": "diffusion/",
        "min_delta": 1e-3,
        "validation_kwargs": {},
    },

    "flow_matching": {
        "method_name": "flow_matching",
        "method": "models.generative.FlowMatching",
        "horizon_length": 31,
        "history_length": 1,
        "stride": 1,
        "model": "models.temporal.TemporalUnet",
        "model_kwargs": {
            "base_hidden_dim": 32,
            "hidden_dim_mult": (1, 2, 4, 8),
            "conv_kernel_size": 5,
            "attention": False,
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

# ------------------------ overrides ------------------------#

fewer_steps = {
    "n_diffusion_steps": 5,
}

one_step = {
    "n_diffusion_steps": 1,
}

long_horizon = {
    "horizon_length": 80,
}

longer_horizon = {
    "horizon_length": 160,
}

data_lim_100 = {
    "train_dataset_size": 100,
}

data_lim_500 = {
    "train_dataset_size": 500,
}

data_lim_1000 = {
    "train_dataset_size": 1000
}

data_lim_2000 = {
    "train_dataset_size": 2000
}

data_lim_3500 = {
    "train_dataset_size": 3500
}

data_lim_5000 = {
    "train_dataset_size": 5000
}

data_lim_10 = {
    "train_dataset_size": 10
}

data_lim_25 = {
    "train_dataset_size": 25
}

data_lim_50 = {
    "train_dataset_size": 50
}

manifold = {
    "manifold": Product(
        input_dim=2,
        manifolds=[
            (FlatTorus(), 1),
            (Euclidean(), 1),
        ],
    ),
    "trajectory_preprocess_fns": [],
    "preprocess_kwargs": {},
    "trajectory_normalizer": None,
    "method_kwargs": {
        "path": "GeodesicProbPath",
        "scheduler": "CondOTScheduler",
        "solver": "RiemannianODESolver",
        "n_fourier_features": 1,
    },
    "min_delta": 5,
}


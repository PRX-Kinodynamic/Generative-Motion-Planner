from os import cpu_count
from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product
import numpy as np
from genMoPlan.utils import watch, handle_angle_wraparound, augment_unwrapped_state_data, watch_dict, process_angles

is_arrakis = False

max_batch_size = int(1e6) if is_arrakis else int(266e3)

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
    "inference": {
        "results_name": watch_dict(results_args_to_watch),
        "attractors": {
            (-2.1, 0): 0,
            (2.1, 0): 0,
            (0, 0): 1,
        },
        "invalid_label": -1,
        "n_runs": 100,
        "batch_size": max_batch_size,
        "attractor_dist_threshold": 0.05,
        "attractor_prob_threshold": 0.98,
        "max_path_length": 502,
        "flow_matching": {
            "n_timesteps": 5,
            "integration_method": "euler",
        },
        "post_process_fns": [
            process_angles,
        ],
        "post_process_fn_kwargs": {
            "angle_indices": [0],
        },
        "final_state_directory": "final_states",
        "generated_trajectory_directory": "generated_trajectories",
    },

    "base": {
        "action_indices": None,
        "angle_indices": [0],
        "loss_type": "l2",
        "loss_weights": None,
        "loss_discount": 1,
        "clip_denoised": False,
        "observation_dim": 2,
        "has_local_query": False,
        "has_global_query": False,

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
        "state_limits": {
            "mins": [-np.pi, -2*np.pi],
            "maxs": [np.pi, 2*np.pi],
        },
        "sample_granularity": 0.04,
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

        "dataset_kwargs": {
            "cost_mul_threshold": 1.0,
        },

        #---------------------------- training ----------------------------#
        "num_epochs": 100,
        "min_num_batches_per_epoch": 1e4,
        "save_freq": 20, # epochs
        "log_freq": 1e3, # steps
        "batch_size": 64,
        "num_workers": 4,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.995,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": 42,

        #---------------------------- validation ----------------------------#
        "val_dataset_size": 40,
        "val_batch_size": 2048,
        "patience": 10,
        "early_stopping": False,
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
        "manifold": None,
    },

    "flow_matching": {
        "method_name": "flow_matching",
        "method": "models.generative.FlowMatching",
        "manifold": None,
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
    "trajectory_normalizer": "LimitsNormalizer",
    "plan_normalizer": None,
    "normalizer_params": {
        "trajectory": {
            "mins": [None, -2*np.pi],
            "maxs": [None, 2*np.pi],
        },
        "plan": None,
    },
    "method_kwargs": {
        "path": "GeodesicProbPath",
        "scheduler": "CondOTScheduler",
        "solver": "RiemannianODESolver",
        "n_fourier_features": 1,
    },
    "min_delta": 5,
}

adaptive_training = {
    "prefix": "adaptive_training/",
    "num_epochs": 20,
    "min_num_batches_per_epoch": 3e3,
    "save_freq": 20, # epochs
    "log_freq": 1e3, # steps
    "batch_size": 64,
    "num_workers": 4,
    "learning_rate": 2e-4,
    "gradient_accumulate_every": 1,
    "ema_decay": 0.995,
    "save_parallel": False,
    "n_reference": 8,
    "bucket": None,
    "device": "cuda",
    "seed": 42,
    "min_delta": 0.0001,
    "patience": 7,
    "early_stopping": True,
    "adaptive_training_kwargs": {
        "combiner": "adaptive_training.ConcatCombiner",
        "combiner_kwargs": {},
        # "uncertainty": "adaptive_training.FinalStateStd",
        # "stop_uncertainty": 0.01,
        "uncertainty": "adaptive_training.FinalStateVariance",
        "stop_uncertainty": 0.001,
        "animate_plots": True,
        "uncertainty_kwargs": {
            "n_runs": 10,
            "device": "cuda",
            "angle_indices": [0],
            "batch_size": max_batch_size,
            "inference_normalization_params": {
                "mins": [-np.pi, -2*np.pi],
                "maxs": [np.pi, 2*np.pi],
            },
            "conditional_sample_kwargs": {
                "n_timesteps": 5,
                "integration_method": "euler",
            },
            "post_process_fns": [
                process_angles,
            ],
            "post_process_fn_kwargs": {
                "angle_indices": [0],
            },
            "num_inference_steps": 17,
        },
        "sampler": "adaptive_training.WeightedDiscreteSampler",
        "sampler_kwargs": {},
        "init_size": 10,
        "step_size": 10,
        "val_size": 40,
        "max_iters": 30,
        "filter_seen": False,
    },
}
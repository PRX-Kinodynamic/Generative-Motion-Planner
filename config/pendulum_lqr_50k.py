import socket
from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product
import numpy as np
from genMoPlan.utils import watch, handle_angle_wraparound, augment_unwrapped_state_data, watch_dict, process_angles, get_experiments_path, shift_to_zero_center_angles

is_arrakis = 'arrakis' in socket.gethostname()

max_batch_size = int(1e6) if is_arrakis else int(266e3)

def read_trajectory(sequence_path):
    with open(sequence_path, "r") as f:
        lines = f.readlines()

    trajectory = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line == "":
            if i < len(lines) - 1:
                raise ValueError(f"[ config/pendulum_lqr_50k ] Empty line found at {sequence_path} at line {i}")
            else:
                break

        state = line.split(',')

        state = [s for s in state if s != ""]

        if len(state) < 2:
            raise ValueError(f"[ config/pendulum_lqr_50k ] Trajectory at {sequence_path} has less than 2 states at line {i}")

        state = state[:2]

        state = [float(s) for s in state]

        trajectory.append(state)

    return np.array(trajectory, dtype=np.float32)

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

logbase = get_experiments_path()

base = {
    "inference": {
        "results_name": watch_dict(results_args_to_watch),
        "attractors": {
            (-2.1, 0): 0,
            (2.1, 0): 0,
            (0, 0): 1,
        },
        "invalid_label": -1,
        "n_runs": 10,
        "batch_size": max_batch_size,
        "attractor_dist_threshold": 0.075,
        "attractor_prob_threshold": 0.6,
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
        "manifold_unwrap_fns": [shift_to_zero_center_angles],
        "manifold_unwrap_kwargs": {
            "angle_indices": [0],
        },
        "load_ema": True,
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
        "read_trajectory_fn": read_trajectory,
        "trajectory_normalizer": "LimitsNormalizer",
        "plan_normalizer": None,
        "normalizer_params": {
            "trajectory": {
                "mins": [-2*np.pi, -2*np.pi],
                "maxs": [2*np.pi, 2*np.pi],
            },
            "plan": None,
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
        "min_num_steps_per_epoch": 0,
        "save_freq": 20, # epochs
        "log_freq": 1e2, # steps
        "batch_size": 1024,
        "num_workers": 4,
        "learning_rate": 1e-4,
        "use_lr_scheduler": True,
        "lr_scheduler_warmup_steps": 1500,
        "lr_scheduler_min_lr": 2e-5,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.995,
        "save_parallel": False,
        "device": "cuda",
        "seed": 42,

        #---------------------------- early stopping-------------------------#
        "patience": 10,
        "warmup_epochs": 5,
        "early_stopping": False,

        #---------------------------- validation ----------------------------#
        "val_dataset_size": 100,
        "val_batch_size": max_batch_size,
        "val_seed": 42,
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
        "min_delta": 1e-2,
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

data_lim_10 = {
    "train_dataset_size": 10,
    "num_epochs": 200000,
}

data_lim_25 = {
    "train_dataset_size": 25,
    "num_epochs": 80000,
}

data_lim_50 = {
    "train_dataset_size": 50,
    "num_epochs": 40000,
}

data_lim_100 = {
    "train_dataset_size": 100,
    "num_epochs": 20000,
}

data_lim_500 = {
    "train_dataset_size": 500,
    "num_epochs": 4000,
}

data_lim_1000 = {
    "train_dataset_size": 1000,
    "num_epochs": 2000,
}

data_lim_2000 = {
    "train_dataset_size": 2000,
    "num_epochs": 1000,
}

data_lim_3500 = {
    "train_dataset_size": 3500,
    "num_epochs": 550,
}

data_lim_5000 = {
    "train_dataset_size": 5000,
    "num_epochs": 400,
}



manifold = {
    "manifold": Product(
        input_dim=2,
        manifolds=[
            (FlatTorus(), 1),
            (Euclidean(), 1),
        ],
    ),
    "manifold_unwrap_fns": [shift_to_zero_center_angles],
    "manifold_unwrap_kwargs": {
        "angle_indices": [0],
    },
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
}

adaptive_training = {
    "prefix": "adaptive_training/",
    "num_epochs": 20,
    "device": "cuda",
    "seed": 42,
    "min_delta": 1e-2,
    "patience": 10,
    "warmup_epochs": 5,
    "early_stopping": True,
    "adaptive_training_kwargs": {
        "n_runs": 10,
        "num_inference_steps": 17,
        "sampling_batch_size": max_batch_size,
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
        "combiner": "adaptive_training.ConcatCombiner",
        "combiner_kwargs": {},
        "animate_plots": True,
        "uncertainty_kwargs": {
            "inference_normalization_params": {
                "mins": [-np.pi, -2*np.pi],
                "maxs": [np.pi, 2*np.pi],
            },
        },
        "sampler": "adaptive_training.WeightedDiscreteSampler",
        "sampler_kwargs": {},
        "init_size": 20,
        "step_size": 20,
        "val_size": 40,
        "max_iters": 30,
        "filter_seen": False,
    },
}

small_dataset = {
    "adaptive_training_kwargs": {
        "init_size": 10,
        "step_size": 10,
    }
}

uncertainty_variance = {
    "adaptive_training_kwargs": {
        "uncertainty": "adaptive_training.FinalStateVariance",
        "stop_uncertainty": 0.001,
    }
}
uncertainty_std = {
    "adaptive_training_kwargs": {
        "uncertainty": "adaptive_training.FinalStateStd",
        "stop_uncertainty": 0.01,
    }
}

adaptive_training_test = {
    "adaptive_training_kwargs": {
        "n_runs": 10,
        "num_inference_steps": 17,
        "max_iters": 1,
    }
}
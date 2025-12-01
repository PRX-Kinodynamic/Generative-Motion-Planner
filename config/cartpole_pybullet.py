import socket
from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product, Sphere
import numpy as np

from genMoPlan.utils import watch, watch_dict, shift_to_zero_center_angles, handle_angle_wraparound, augment_unwrapped_state_data, process_angles, get_experiments_path

is_arrakis = 'arrakis' in socket.gethostname()

max_batch_size = int(2.5e5) if is_arrakis else int(1e4)


max_path_length = 613

state_names = ["x", "theta", "x_dot", "theta_dot"]

# Based on initial_state_bounds from dataset_description.json
mins = [-6.0, -np.pi, -5.0, -5.0]
maxs = [6.0, np.pi, 5.0, 5.0]

manifold_mins = mins.copy()
manifold_maxs = maxs.copy()

manifold_mins[1] = None
manifold_maxs[1] = None

manifold = Product(
    input_dim=4,
    manifolds=[
        (Euclidean(), 1),
        (FlatTorus(), 1),
        (Euclidean(), 2)
    ],
)

angle_indices = [1]

def attractor_classification_fn(final_states: np.ndarray, attractors: dict, attractor_dist_threshold: float, invalid_label: int = -1, verbose: bool = True):
    if verbose:
        print("[ config/cartpole_pybullet ] Getting attractor labels for trajectories")

    final_norms = np.linalg.norm(final_states, axis=1)
    predicted_labels = np.zeros_like(final_norms)

    predicted_labels[final_norms <= attractor_dist_threshold] = 1
    predicted_labels[final_norms > attractor_dist_threshold] = 0

    return predicted_labels

def read_trajectory(sequence_path):
    with open(sequence_path, "r") as f:
        lines = f.readlines()

    trajectory = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line == "":
            if i < len(lines) - 1:
                raise ValueError(f"[ config/cartpole_pybullet ] Empty line found at {sequence_path} at line {i}")
            else:
                break

        state = line.split(',')

        state = [s for s in state if s != ""]

        if len(state) < 4:
            raise ValueError(f"[ config/cartpole_pybullet ] Trajectory at {sequence_path} has less than 4 states at line {i}")

        state = state[:4]

        state = np.array([float(s) for s in state])

        state[1] = state[1] % (2 * np.pi) # wrap angle to [0, 2pi]
        if state[1] > np.pi:
            state[1] -= 2 * np.pi

        trajectory.append(state)

    return np.array(trajectory, dtype=np.float32)

def ground_truth_filter_fn(start_states: np.ndarray, expected_labels: np.ndarray, model_args):
    train_dataset_size = model_args.train_dataset_size
    return start_states[train_dataset_size:], expected_labels[train_dataset_size:]


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
        "attractor_classification_fn": attractor_classification_fn,
        "invalid_label": -1,
        "n_runs": 20,
        "batch_size": max_batch_size,
        "attractor_dist_threshold": 1,
        "attractor_prob_threshold": 0.6,
        "flow_matching": {
            "n_timesteps": 10,
            "integration_method": "euler",
        },
        "post_process_fns": [
            process_angles,
        ],
        "post_process_fn_kwargs": {
            "angle_indices": angle_indices,
        },
        "final_state_directory": "final_states",
        "generated_trajectory_directory": "generated_trajectories",
        "manifold_unwrap_fns": [shift_to_zero_center_angles],
        "manifold_unwrap_kwargs": {
            "angle_indices": angle_indices,
        },
        "load_ema": True,
        # "ground_truth_filter_fn": ground_truth_filter_fn,
        "attractor_labels": [0, 1],
        "max_path_length": max_path_length,
    },

    "base": {
        "action_indices": None,
        "angle_indices": angle_indices,
        "loss_type": "l2",
        "clip_denoised": False,
        "observation_dim": 4,
        "has_local_query": False,
        "has_global_query": False,
        "state_names": state_names,
        "max_path_length": max_path_length,

        #-------------------------------- dataset --------------------------------#
        "loader": "datasets.TrajectoryDataset",
        "read_trajectory_fn": read_trajectory,
        "trajectory_normalizer": "LimitsNormalizer",
        "plan_normalizer": None,
        "normalizer_params": {
            "trajectory": {
                "mins": mins,
                "maxs": maxs,
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
                "angle_indices": angle_indices,
            },
            "plan": None,
        },
        "use_history_padding": False,
        "use_horizon_padding": False,
        "use_history_mask": False,
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
        "useAdamW": False,
        "optimizer_kwargs": {},
        "use_lr_scheduler": True,
        "lr_scheduler_warmup_steps": 1500,
        "lr_scheduler_min_lr": 2e-5,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.995,
        "save_parallel": False,
        "device": "cuda",
        "seed": 42,
        "clip_grad_norm": None,

        #---------------------------- early stopping-------------------------#
        "patience": 10,
        "warmup_epochs": 5,
        "early_stopping": False,

        #---------------------------- validation ----------------------------#
        "val_dataset_size": 100,
        "val_batch_size": max_batch_size,
        "val_seed": 42,

        #-------------------------------evaluation--------------------------#
        "perform_final_state_evaluation": True,
        "eval_freq": 10, # epochs
        "eval_batch_size": max_batch_size,
        "eval_seed": 42,
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
        "manifold": manifold,
        "manifold_unwrap_fns": [],
        "manifold_unwrap_kwargs": {
            "angle_indices": angle_indices,
        },
        "trajectory_preprocess_fns": [],
        "preprocess_kwargs": {},
        "normalizer_params": {
            "trajectory": {
                "mins": manifold_mins,
                "maxs": manifold_maxs,
            },
            "plan": None,
        },
        "horizon_length": 15,
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
            "path": "GeodesicProbPath",
            "scheduler": "CondOTScheduler",
            "solver": "RiemannianODESolver",
            "n_fourier_features": 1,
        },
        "prefix": "flow_matching/",
        "min_delta": 1e-2,
        "validation_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
            "max_path_length": max_path_length,
        },
    }
}

data_lim_10 = {
    "train_dataset_size": 10,
    "num_epochs": 200000,
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

data_lim_20000 = {
    "train_dataset_size": 20000,
    "num_epochs": 2000,
}

larger_hidden_dim = {
    "model_kwargs": {
        "base_hidden_dim": 64,
    },
}

lr_test = {
    "learning_rate": 1e-3,
    "lr_scheduler_warmup_steps": 300,
    "lr_scheduler_min_lr": 2e-4,
}

dit_test = {
    "model": "models.temporal.TemporalDiffusionTransformer",
    "model_kwargs": {
        "hidden_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "feedforward_dim": None,
        "dropout": 0.01,
        "time_embed_dim": None,
        "global_query_embed_dim": None,
        "local_query_embed_dim": None,
        "use_positional_encoding": True,
    },

    "lr_scheduler_warmup_steps": 500,
    "learning_rate": 2.5e-4,
    "lr_scheduler_min_lr": 2e-5,
    "ema_decay": 0.999,
    "useAdamW": True,
    "optimizer_kwargs": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.02,
    },
    "clip_grad_norm": 1.0,

    "val_batch_size": int(6e4) if is_arrakis else int(1e4),
}

stride_10 = {
    "stride": 10,
}

stride_30 = {
    "stride": 30,
}

epochs_1000 = {
    "num_epochs": 1000,
}

epochs_500 = {
    "num_epochs": 500,
}

next_history_loss_weight = {
    "loss_weight_type": "next_history",
    "loss_weight_kwargs": {
        "lambda_next_history": 2.0,
    },
}

next_history_loss_weight_5 = {
    "loss_weight_type": "next_history",
    "loss_weight_kwargs": {
        "lambda_next_history": 5.0,
    },
}

history_mask_2 = {
    "use_history_mask": True,
    "use_horizon_padding": False,
    "history_length": 2,
    "final_state_evaluation": False,
}

history_padding_2 = {
    "use_history_padding": True,
    "use_history_mask": False,
    "use_horizon_padding": False,
    "history_length": 2,
    "final_state_evaluation": False,
}

history_mask_5 = {
    "use_history_mask": True,
    "use_horizon_padding": False,
    "history_length": 5,
    "final_state_evaluation": False,
}

history_padding_5 = {
    "use_history_padding": True,
    "use_history_mask": False,
    "use_horizon_padding": False,
    "history_length": 5,
    "final_state_evaluation": False,
}

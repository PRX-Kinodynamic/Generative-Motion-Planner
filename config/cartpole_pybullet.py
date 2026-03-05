import socket
from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product, Sphere
import numpy as np

from genMoPlan.utils import watch, watch_dict, shift_to_zero_center_angles, handle_angle_wraparound, augment_unwrapped_state_data, process_angles, get_experiments_path

is_arrakis = 'arrakis' in socket.gethostname()

max_batch_size = int(2.5e5) if is_arrakis else int(1e4)


max_path_length = 613
# new change to experiment with smaller path length with smaller strides
# max_path_length = 150

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
            # Rollout loss dataset configuration (must match method_kwargs.rollout_steps)
            # "rollout_steps": 2,  # Must match method_kwargs.rollout_steps
            # "rollout_target_mode": "gt_future",  # How to generate rollout targets
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
        #---------------------------- optional initialization ----------------------------#
        # No-op defaults: existing runs remain unchanged unless init_from is set.
        "init_from": None,  # Directory containing checkpoint (e.g., /path/to/stage1/run)
        "init_state_name": "best.pt",
        "init_use_ema": True,
        "init_strict": False,
        "init_reset_optimizer": True,
        "init_reset_scheduler": True,

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
        "eval_freq": 1,  # epochs - controls verbose validation output frequency
        "detailed_eval_freq": 10,  # epochs - controls detailed evaluation (Final State, Sequential, Full Traj). 0=disabled
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
            # Rollout loss configuration (optional)
            # "use_rollout_loss": True,
            # "rollout_steps": 2,  # Number of rollout steps (k=1..K)
            # "rollout_weighting": [1.0, 0.5],  # Weights for each rollout step, or dict with schedule
            # "rollout_loss_type": "l2",  # "l2" or "manifold"
            # "rollout_sample_kwargs": {
            #     "n_timesteps": 5,
            #     "integration_method": "euler",
            #     "batch_frac": 1.0,  # Fraction of batch to use for rollout (to control compute)
            # },
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
max_path_572 = {
    "max_path_length": 572,
    "validation_kwargs": {
        "max_path_length": 572,
    },
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

stride_2 = {
    "stride": 2,
}

stride_3 = {
    "stride": 3,
}

stride_5 = {
    "stride": 5,
}

stride_10 = {
    "stride": 10,
}

# Combined variations with smaller strides for better swing-up coverage
stride_2_horizon_15 = {
    "stride": 2,
    "horizon_length": 15,
}

stride_3_horizon_15 = {
    "stride": 3,
    "horizon_length": 15,
}

stride_30 = {
    "stride": 30,
}

data_lim_200 = {
    "train_dataset_size": 200,
    "num_epochs": 2000,
}

# Max path length variations (updates both base and validation_kwargs)
path_length_150 = {
    "max_path_length": 150,
    "validation_kwargs": {
        "max_path_length": 150,
    },
}

path_length_300 = {
    "max_path_length": 300,
    "validation_kwargs": {
        "max_path_length": 300,
    },
}

path_length_450 = {
    "max_path_length": 450,
    "validation_kwargs": {
        "max_path_length": 450,
    },
}

path_length_613 = {
    "max_path_length": 613,
    "validation_kwargs": {
        "max_path_length": 613,
    },
}

horizon_7 = {
    "horizon_length": 7,
}

horizon_15 = {
    "horizon_length": 15,
}

horizon_31 = {
    "horizon_length": 31,
}

# Stride 1 with short path - maximum temporal resolution
stride_1_horizon_7_path_150 = {
    "stride": 1,
    "horizon_length": 7,
    "max_path_length": 150,
    "validation_kwargs": {
        "max_path_length": 150,
    },
}

stride_1_horizon_15_path_150 = {
    "stride": 1,
    "horizon_length": 15,
    "max_path_length": 150,
    "validation_kwargs": {
        "max_path_length": 150,
    },
}

stride_2_horizon_7_path_150 = {
    "stride": 2,
    "horizon_length": 7,
    "max_path_length": 150,
    "validation_kwargs": {
        "max_path_length": 150,
    },
}

stride_2_horizon_15_path_300 = {
    "stride": 2,
    "horizon_length": 15,
    "max_path_length": 150,
    "validation_kwargs": {
        "max_path_length": 300
    },
}

stride_3_horizon_15_path_200 = {
    "stride": 3,
    "horizon_length": 15,
    "max_path_length": 200,
    "validation_kwargs": {
        "max_path_length": 200,
    },
}

stride_3_horizon_15_path_300 = {
    "stride": 3,
    "horizon_length": 15,
    "max_path_length": 300,
    "batch_size": 4000,
    "validation_kwargs": {
        "max_path_length": 300,
    },
}

# Stride 1 with long path - maximum temporal resolution
stride_1_horizon_15_path_300 = {
    "stride": 1,
    "horizon_length": 15,
    "max_path_length": 300,
    "batch_size": 4000,
    "validation_kwargs": {
        "max_path_length": 300,
    },
}

# Combined variations with horizon_31 (total sequence=32, works with UNet)
stride_2_horizon_31 = {
    "stride": 2,
    "horizon_length": 31,
}

stride_3_horizon_31 = {
    "stride": 3,
    "horizon_length": 31,
}

stride_5_horizon_31 = {
    "stride": 5,
    "horizon_length": 31,
}

# Single-step inference: stride=19 with horizon=31 gives actual_horizon=571
# This allows single-step rollout for max_path_length=572
stride_19_horizon_31_single_step = {
    "stride": 19,
    "horizon_length": 31,
    "use_horizon_padding": True,  # Allow shorter validation trajectories
}

stride_25_horizon_31_single_step = {
    "stride": 25,
    "horizon_length": 31,
    "use_horizon_padding": True,  # Allow shorter validation trajectories
}

# Combined variation for quick swing-up learning experiments
# horizon_length=7 gives total sequence=8 (works with UNet)
stride_5_horizon_7 = {
    "stride": 5,
    "horizon_length": 7,
}

# Alternative with original horizon (total sequence=16)
stride_5_horizon_15 = {
    "stride": 5,
    "horizon_length": 15,
}

epochs_2000 = {
    "num_epochs": 2000,
}

epochs_1000 = {
    "num_epochs": 1000,
}

epochs_500 = {
    "num_epochs": 500,
}

epochs_100 = {
    "num_epochs": 100,
}

epochs_25 = {
    "num_epochs": 25,
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

# Rollout loss configurations
rollout_loss_2_steps = {
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_steps": 2,
        "rollout_weighting": [1.0],  # Weight for step 1
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
            "batch_frac": 1.0,
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 2,
        "rollout_target_mode": "gt_future",
    },
}

rollout_loss_3_steps = {
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_steps": 3,
        "rollout_weighting": [1.0, 0.5],  # Weights for steps 1 and 2
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
            "batch_frac": 1.0,
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 3,
        "rollout_target_mode": "gt_future",
    },
}

rollout_loss_exp_decay = {
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_steps": 3,
        "rollout_weighting": {
            "type": "exp",
            "decay": 0.5,  # Each step gets 0.5x weight of previous
        },
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
            "batch_frac": 0.5,  # Use 50% of batch to control compute
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 3,
        "rollout_target_mode": "gt_future",
    },
}

# Improved rollout loss with 3 steps and exponential decay (recommended)
rollout_loss_3_steps_exp = {
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_steps": 3,
        "rollout_weighting": {
            "type": "exp",
            "decay": 0.7,  # Each step gets 0.7x weight (less aggressive decay)
        },
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
            "batch_frac": 1.0,  # Use full batch for better training
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 3,
        "rollout_target_mode": "gt_future",
    },
}

# Adaptive autoregressive rollout (shared anchor/prediction shifts).
# Note: rollout_steps follows existing semantics (dataset emits rollout_steps - 1 targets),
# so rollout_steps=4 gives three adaptive targets.
adaptive_rollout_v1 = {
    "batch_size": 4000,
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_operator": "adaptive_stride",
        "rollout_steps": 4,
        "rollout_weighting": [1.0, 0.6, 0.4],
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
            "batch_frac": 1.0,
        },
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 4,
        "rollout_target_mode": "adaptive_stride",
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
}

# Adaptive rollout with constant shared shift (no increment) and intended for stride=1 runs.
# Use with: stride_1_horizon_15_path_300
adaptive_rollout_stride1_constshift = {
    "batch_size": 4000,
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_operator": "adaptive_stride",
        "rollout_steps": 4,
        "rollout_weighting": [1.0, 0.6, 0.4],
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
            "batch_frac": 1.0,
        },
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            # Constant shift of +1 sampled step between re-anchors/target windows.
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 1,
                "delta": 0,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 4,
        "rollout_target_mode": "adaptive_stride",
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 1,
                "delta": 0,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
}

adaptive_rollout_step1_only = {
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_operator": "adaptive_stride",
        "rollout_steps": 4,
        "rollout_weighting": [1.0, 0.0, 0.0],
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
            "batch_frac": 1.0,
        },
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 4,
        "rollout_target_mode": "adaptive_stride",
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
}

adaptive_rollout_step2_only = {
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_operator": "adaptive_stride",
        "rollout_steps": 4,
        "rollout_weighting": [0.0, 1.0, 0.0],
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
            "batch_frac": 1.0,
        },
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 4,
        "rollout_target_mode": "adaptive_stride",
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
}

adaptive_rollout_step3_only = {
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_operator": "adaptive_stride",
        "rollout_steps": 4,
        "rollout_weighting": [0.0, 0.0, 1.0],
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
            "batch_frac": 1.0,
        },
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 4,
        "rollout_target_mode": "adaptive_stride",
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
}

# ---------------------------- Two-stage training schemes ----------------------------#
# Stage 1: fast base FM pretrain (no rollout loss).
stage1_base_fast = {
    "method_kwargs": {
        "use_rollout_loss": False,
        "rollout_steps": 1,
    },
    "dataset_kwargs": {
        "rollout_steps": 1,
    },
    # Enable extended reporting every N epochs (0 disables it).
    "detailed_eval_freq": 10,
}

# Stage 2: adaptive rollout fine-tuning from a Stage 1 checkpoint.
# Provide --init_from on CLI; all init_* defaults are already in base.
stage2_adaptive_ft = {
    "learning_rate": 5e-5,  # Lower LR for fine-tuning (half of base 1e-4)
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_operator": "adaptive_stride",
        "rollout_steps": 4,
        "rollout_weighting": [1.0, 0.6, 0.4],
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 5,  # Increased from 3 for better sampling quality
            "integration_method": "euler",
            "batch_frac": 0.5,  # Increased from 0.25 to 0.5 for more rollout loss signal
        },
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 4,
        "rollout_target_mode": "adaptive_stride",
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
    # Enable extended reporting every N epochs (0 disables it).
    "detailed_eval_freq": 10,
}

# Stage-2 ablation: only first adaptive rollout step contributes to loss.
stage2_adaptive_step1_ft = {
    "method_kwargs": {
        "use_rollout_loss": True,
        "rollout_operator": "adaptive_stride",
        "rollout_steps": 4,
        "rollout_weighting": [1.0, 0.0, 0.0],
        "rollout_loss_type": "l2",
        "rollout_sample_kwargs": {
            "n_timesteps": 3,
            "integration_method": "euler",
            "batch_frac": 0.25,
        },
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
    "dataset_kwargs": {
        "rollout_steps": 4,
        "rollout_target_mode": "adaptive_stride",
        "adaptive_rollout": {
            "enabled": True,
            "mode": "adaptive_stride",
            "shared_shift_schedule": {
                "type": "arithmetic",
                "start": 2,
                "delta": 1,
            },
            "base_span": 15,
            "max_span": 15,
        },
    },
    # Enable extended reporting every N epochs (0 disables it).
    "detailed_eval_freq": 10,
}

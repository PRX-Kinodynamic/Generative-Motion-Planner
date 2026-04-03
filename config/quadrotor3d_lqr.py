"""
Configuration for 3D Quadrotor LQR environment (PyBullet).

This config contains training/model setup. System-specific details
(state limits, manifolds, preprocessing) are provided by Quadrotor3DLQRSystem.

Dataset: quadrotor3D_lqr
State: [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r]
Goal: Stabilize to hover at (0, 0, 1) with identity orientation
Controller: LQR
"""

import socket

from genMoPlan.utils import watch, watch_dict, get_experiments_path
from genMoPlan.systems import Quadrotor3DLQRSystem

is_westeros = "westeros" in socket.gethostname()
max_batch_size = int(1e4) if is_westeros else int(2.5e5)


# -------------------------------- System -------------------------------- #


def get_system(config=None, use_manifold: bool = False, dataset: str = None, **kwargs):
    """
    Create a Quadrotor3DLQRSystem from this config.

    Args:
        config: Optional config dict override. If None, uses the base config.
        use_manifold: Whether to use manifold-based flow matching.
        dataset: Name of the dataset (required for loading achieved bounds).
        **kwargs: Additional arguments to override system parameters.

    Returns:
        Quadrotor3DLQRSystem instance.
    """
    if config is None:
        config = base

    # Always use the correct dataset directory name (case-sensitive)
    dataset = "quadrotor3D_lqr"

    method_config = config.get("flow_matching", config.get("diffusion", {}))
    return Quadrotor3DLQRSystem(
        dataset=dataset,
        stride=kwargs.get("stride", method_config.get("stride", 1)),
        history_length=kwargs.get(
            "history_length", method_config.get("history_length", 1)
        ),
        horizon_length=kwargs.get(
            "horizon_length", method_config.get("horizon_length", 31)
        ),
        use_manifold=use_manifold,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["stride", "history_length", "horizon_length", "dataset"]
        },
    )


# -------------------------------- Experiment naming -------------------------------- #

exp_args_to_watch = [
    ("history_length", "HILEN"),
    ("horizon_length", "HOLEN"),
    ("use_history_padding", "HIPAD"),
    ("use_horizon_padding", "HOPAD"),
    ("stride", "STRD"),
]

results_args_to_watch = [
    ("n_runs", "NRUN"),
    ("outcome_prob_threshold", "OPTH"),
]

logbase = get_experiments_path()


# -------------------------------- Base config -------------------------------- #

base = {
    "inference": {
        "results_name": watch_dict(results_args_to_watch),
        "n_runs": 20,
        "batch_size": max_batch_size,
        "outcome_prob_threshold": 0.6,
        "flow_matching": {
            "n_timesteps": 10,
            "integration_method": "euler",
        },
        "final_state_directory": "final_states/eval",
        "generated_trajectory_directory": "generated_trajectories",
        "load_ema": True,
        "inference_mask_strategy": "first_step_only",
    },
    "base": {
        "action_indices": None,
        "loss_type": "l2",
        "clip_denoised": False,
        "has_local_query": False,
        "has_global_query": True,
        # -------------------------------- dataset --------------------------------#
        "loader": "datasets.TrajectoryDataset",
        "shuffled_indices_fname": "all_shuffled_indices.txt",
        "use_history_padding": False,
        "use_horizon_padding": True,
        "use_history_mask": False,
        "use_plan": False,
        "train_dataset_size": 25000,
        "is_history_conditioned": False,
        # ---------------------------- serialization ----------------------------#
        "logbase": logbase,
        "exp_name": watch(exp_args_to_watch),
        "dataset_kwargs": {
            "cost_mul_threshold": 1.0,
        },
        # ---------------------------- training ----------------------------#
        "num_epochs": 2000,
        "min_num_steps_per_epoch": 0,
        "save_freq": 20,  # epochs
        "log_freq": 1e2,  # steps
        "batch_size": 1024,
        "num_workers": 4,
        "learning_rate": 2e-4,
        "useAdamW": True,
        "optimizer_kwargs": {
            "betas": (0.9, 0.95),
            "weight_decay": 0.02,
        },
        "use_lr_scheduler": True,
        "lr_scheduler_warmup_steps": 1000,
        "lr_scheduler_min_lr": 2e-5,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.999,
        "save_parallel": False,
        "device": "cuda",
        "seed": 42,
        "clip_grad_norm": 1.0,
        # ---------------------------- early stopping-------------------------#
        "patience": 10,
        "warmup_epochs": 5,
        "early_stopping": False,
        # ---------------------------- validation ----------------------------#
        "val_dataset_size": 100,
        "val_batch_size": int(1e4) if is_westeros else int(6e4),
        "val_seed": 42,
        # -------------------------------evaluation--------------------------#
        "perform_final_state_evaluation": True,
        "eval_freq": 10,  # epochs
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
            # Scaled up for 13D state space
            "base_hidden_dim": 64,
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
        "model": "models.temporal.TemporalDiffusionTransformer",
        "model_kwargs": {
            # DiT sized for 13D state space
            "hidden_dim": 256,
            "num_layers": 8,
            "num_heads": 8,
            "feedforward_dim": None,
            "dropout": 0.01,
            "time_embed_dim": None,
            "local_query_embed_dim": None,
            "query_encoder_layers": 1,
        },
        "method_kwargs": {
            "path": "AffineProbPath",
            "scheduler": "CondOTScheduler",
            "solver": "ODESolver",
            "n_fourier_features": 1,
        },
        "prefix": "flow_matching/",
        "min_delta": 1e-2,
        "validation_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
        },
        "use_manifold": False,
    },
}


# -------------------------------- Overrides -------------------------------- #
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

data_lim_5000 = {
    "train_dataset_size": 5000,
    "num_epochs": 1000,
}

data_lim_10000 = {
    "train_dataset_size": 10000,
    "num_epochs": 500,
}

data_lim_14000 = {
    "train_dataset_size": 14000,
    "num_epochs": 2000,
    "early_stopping": False,
}


data_lim_25000 = {
    "train_dataset_size": 25000,
    "num_epochs": 2000,
}

smaller_hidden_dim = {
    "model_kwargs": {
        "base_hidden_dim": 32,
    },
}

larger_hidden_dim = {
    "model_kwargs": {
        "base_hidden_dim": 128,
    },
}

lr_test = {
    "learning_rate": 1e-3,
    "lr_scheduler_warmup_steps": 300,
    "lr_scheduler_min_lr": 2e-4,
}

dit_test = {
    # DiT sized for 13D state space
    "model": "models.temporal.TemporalDiffusionTransformer",
    "model_kwargs": {
        "hidden_dim": 256,
        "num_layers": 8,
        "num_heads": 8,
        "feedforward_dim": None,
        "dropout": 0.01,
        "time_embed_dim": None,
        "local_query_embed_dim": None,
    },
    "lr_scheduler_warmup_steps": 1000,
    "learning_rate": 2e-4,
    "lr_scheduler_min_lr": 2e-5,
    "ema_decay": 0.999,
    "useAdamW": True,
    "optimizer_kwargs": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.02,
    },
    "clip_grad_norm": 1.0,
    "val_batch_size": int(1e4) if is_westeros else int(6e4),
}

no_manifold = {
    "use_manifold": False,
    "method_kwargs": {
        "path": "AffineProbPath",
        "solver": "ODESolver",
    },
}

manifold = {
    "use_manifold": True,
    "method_kwargs": {
        "path": "GeodesicProbPath",
        "solver": "RiemannianODESolver",
    },
}

unet = {
    "model": "models.temporal.TemporalUnet",
    "model_kwargs": {
        "base_hidden_dim": 64,
        "hidden_dim_mult": (1, 2, 4, 8),
        "conv_kernel_size": 5,
        "attention": False,
    },
    "useAdamW": False,
    "learning_rate": 1e-4,
    "lr_scheduler_warmup_steps": 1500,
    "ema_decay": 0.995,
    "optimizer_kwargs": {},
    "clip_grad_norm": None,
    "val_batch_size": max_batch_size,
}

history_as_condition = {
    "is_history_conditioned": True,
    "has_global_query": False,
}

history_as_query = {
    "is_history_conditioned": False,
    "has_global_query": True,
    "use_history_mask": False,
    "model_kwargs": {
        "query_encoder_layers": 1,
    },
}

stride_10 = {
    "stride": 10,
}

stride_25 = {
    "stride": 25,
}

epochs_1000 = {
    "num_epochs": 1000,
}

epochs_500 = {
    "num_epochs": 500,
}

single_horizon = {
    # stride=22: actual_horz = 30*22+1 = 661 >= 635 = max_path_length-1
    # So ceil(635/661) = 1 inference step (single horizon)
    "stride": 22,
}

one_horizon_1_state_pred = {
    "stride": 635,
    "horizon_length": 1,
}

one_horizon_2_state_pred = {
    "stride": 318,
    "horizon_length": 2,
}

one_horizon_10_state_pred = {
    "stride": 64,
    "horizon_length": 10,
}

one_horizon_20_state_pred = {
    "stride": 32,
    "horizon_length": 20,
}

one_horizon_31_state_pred = {
    "stride": 21,
    "horizon_length": 31,
}

history_as_query = {
    "is_history_conditioned": False,
    "has_global_query": True,
    "use_history_mask": False,
    "model_kwargs": {
        "query_encoder_layers": 1,
    },
}

final_state_quick_eval = {
    "eval_freq": 1,
}

"""
Configuration for Cartpole PyBullet environment.

This config contains training/model setup. System-specific details
(state limits, manifolds, preprocessing) are provided by CartpolePyBulletSystem.
"""
import socket

from genMoPlan.utils import watch, watch_dict, get_experiments_path
from genMoPlan.systems import CartpolePyBulletSystem

is_arrakis = "arrakis" in socket.gethostname()
max_batch_size = int(2.5e5) if is_arrakis else int(1e4)


# -------------------------------- System -------------------------------- #

# Create default system for backward compatibility with scripts that import configs directly
_default_system = CartpolePyBulletSystem.create(stride=1, history_length=1, horizon_length=31)


def get_system(config=None, use_manifold: bool = False, **kwargs):
    """
    Create a CartpolePyBulletSystem from this config.

    Args:
        config: Optional config dict override. If None, uses the base config.
        use_manifold: Whether to use manifold-based flow matching.
        **kwargs: Additional arguments to override system parameters.

    Returns:
        CartpolePyBulletSystem instance.
    """
    if config is None:
        config = base

    method_config = config.get("flow_matching", config.get("diffusion", {}))
    return CartpolePyBulletSystem(
        stride=kwargs.get("stride", method_config.get("stride", 1)),
        history_length=kwargs.get("history_length", method_config.get("history_length", 1)),
        horizon_length=kwargs.get("horizon_length", method_config.get("horizon_length", 31)),
        use_manifold=use_manifold,
        **{k: v for k, v in kwargs.items() if k not in ["stride", "history_length", "horizon_length"]},
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
        "final_state_directory": "final_states",
        "generated_trajectory_directory": "generated_trajectories",
        "load_ema": True,
        # Inference masking strategy (when use_history_mask=True)
        # Options: "first_step_only" (default), "always", "never"
        "inference_mask_strategy": "first_step_only",
    },
    "base": {
        "action_indices": None,
        "loss_type": "l2",
        "clip_denoised": False,
        "has_local_query": False,
        "has_global_query": False,
        # -------------------------------- dataset --------------------------------#
        "loader": "datasets.TrajectoryDataset",
        "plan_normalizer": None,
        "plan_preprocess_fns": None,
        "use_history_padding": False,
        "use_horizon_padding": True,
        "use_history_mask": False,
        "use_plan": False,
        "train_dataset_size": None,
        "is_history_conditioned": True,
        # ---------------------------- serialization ----------------------------#
        "logbase": logbase,
        "exp_name": watch(exp_args_to_watch),
        "dataset_kwargs": {
            "cost_mul_threshold": 1.0,
        },
        # ---------------------------- training ----------------------------#
        "num_epochs": 100,
        "min_num_steps_per_epoch": 0,
        "save_freq": 20,  # epochs
        "log_freq": 1e2,  # steps
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
        # ---------------------------- early stopping-------------------------#
        "patience": 10,
        "warmup_epochs": 5,
        "early_stopping": False,
        # ---------------------------- validation ----------------------------#
        "val_dataset_size": 100,
        "val_batch_size": max_batch_size,
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
        },
        # Set to True to use manifold flow matching (system will provide manifold)
        "use_manifold": True,
    },
}


# -------------------------------- Overrides -------------------------------- #

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

history_mask_zeros = {
    """
    History masking with zero padding and loss weighting.
    - Model explicitly knows which positions are missing (via mask)
    - Missing positions padded with zeros
    - Loss only computed on valid positions
    - Use case: Variable-length demonstrations, neutral padding
    """
    "use_history_mask": True,
    "use_horizon_padding": False,
    "history_length": 5,
    "history_mask_padding_value": "zeros",
    "use_mask_loss_weighting": True,
    "final_state_evaluation": False,
}

history_mask_first = {
    """
    History masking with first-state padding and loss weighting.
    - Model knows which positions are missing
    - Missing positions padded with first available state
    - Loss only computed on valid positions
    - Use case: Systems that start from rest/stationary state
    """
    "use_history_mask": True,
    "use_horizon_padding": False,
    "history_length": 5,
    "history_mask_padding_value": "first",
    "use_mask_loss_weighting": True,
    "final_state_evaluation": False,
}

history_mask_mirror = {
    """
    History masking with mirror padding and loss weighting.
    - Model knows which positions are missing
    - Missing positions filled by reflecting available sequence
    - Loss only computed on valid positions
    - Use case: Smooth/periodic trajectories
    """
    "use_history_mask": True,
    "use_horizon_padding": False,
    "history_length": 5,
    "history_mask_padding_value": "mirror",
    "use_mask_loss_weighting": True,
    "final_state_evaluation": False,
}

history_padding_first = {
    """
    History padding (no mask) with first-state strategy.
    - Model treats all positions as real (no mask)
    - Missing positions padded with first available state
    - Use case: Want model to "imagine" stationary start
    """
    "use_history_padding": True,
    "use_history_mask": False,
    "use_horizon_padding": False,
    "history_length": 5,
    "history_padding_strategy": "first",
    "final_state_evaluation": False,
}

history_padding_mirror = {
    """
    History padding (no mask) with mirror strategy.
    - Model treats all positions as real (no mask)
    - Missing positions filled by mirroring available sequence
    - Use case: Smooth continuation without exposing missing data
    """
    "use_history_padding": True,
    "use_history_mask": False,
    "use_horizon_padding": False,
    "history_length": 5,
    "history_padding_strategy": "mirror",
    "final_state_evaluation": False,
}

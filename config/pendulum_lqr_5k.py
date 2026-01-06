"""
Configuration for Pendulum LQR (5k dataset variant).

This config contains only training/model setup. System-specific details
(state limits, preprocessing) are handled by PendulumLQRSystem.
"""
from genMoPlan.utils import watch, get_experiments_path
from genMoPlan.utils.systems import PendulumLQRSystem


# -------------------------------- System -------------------------------- #

def get_system(config=None, **kwargs):
    """
    Create a PendulumLQRSystem from this config.

    Args:
        config: Optional config dict override. If None, uses the base config.
        **kwargs: Additional arguments to override system parameters.

    Returns:
        PendulumLQRSystem instance.
    """
    if config is None:
        config = base

    method_config = config.get("flow_matching", config.get("diffusion", {}))
    return PendulumLQRSystem(
        name="pendulum_lqr_5k",
        stride=kwargs.get("stride", method_config.get("stride", 1)),
        history_length=kwargs.get("history_length", method_config.get("history_length", 1)),
        horizon_length=kwargs.get("horizon_length", method_config.get("horizon_length", 31)),
        **{k: v for k, v in kwargs.items() if k not in ["stride", "history_length", "horizon_length"]},
    )


# Create default system for extracting system-provided config values
_default_system = PendulumLQRSystem.create(stride=1, history_length=1, horizon_length=31)

# Get system-provided dataset config
_system_dataset_config = _default_system.get_dataset_config(use_manifold=False)
_system_inference_config = _default_system.get_inference_config()


# -------------------------------- Experiment naming -------------------------------- #

args_to_watch = [
    ("history_length", "HILEN"),
    ("horizon_length", "HOLEN"),
    ("use_history_padding", "HIPAD"),
    ("use_horizon_padding", "HOPAD"),
    ("stride", "STRD"),
]

logbase = get_experiments_path()


# -------------------------------- Base config -------------------------------- #

base = {
    "inference": {
        "n_runs": 100,
        "batch_size": int(1e6),
        "outcome_prob_threshold": 0.98,
        "max_path_length": 502,
        "flow_matching": {
            "n_timesteps": 10,
            "integration_method": "euler",
        },
        "final_state_directory": "final_states",
        "generated_trajectory_directory": "generated_trajectories",
        # System-provided inference config
        "post_process_fns": _system_inference_config["post_process_fns"],
        "post_process_fn_kwargs": _system_inference_config["post_process_fn_kwargs"],
        "manifold_unwrap_fns": _system_inference_config["manifold_unwrap_fns"],
        "manifold_unwrap_kwargs": _system_inference_config["manifold_unwrap_kwargs"],
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
        # System-provided dataset config
        "observation_dim": _system_dataset_config["observation_dim"],
        "angle_indices": _system_dataset_config["angle_indices"],
        "state_names": _system_dataset_config["state_names"],
        "max_path_length": _system_dataset_config["max_path_length"],
        "read_trajectory_fn": _system_dataset_config["read_trajectory_fn"],
        "trajectory_normalizer": _system_dataset_config["trajectory_normalizer"],
        "normalizer_params": _system_dataset_config["normalizer_params"],
        "trajectory_preprocess_fns": _system_dataset_config["trajectory_preprocess_fns"],
        "preprocess_kwargs": _system_dataset_config["preprocess_kwargs"],
        # ---------------------------- serialization ----------------------------#
        "logbase": logbase,
        "exp_name": watch(args_to_watch),
        # ---------------------------- training ----------------------------#
        "num_epochs": 100,
        "min_num_steps_per_epoch": 1e4,
        "save_freq": 1e5,
        "log_freq": 1e3,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "useAdamW": False,
        "optimizer_kwargs": {},
        "clip_grad_norm": None,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_parallel": False,
        "device": "cuda",
        "seed": 42,
        # ---------------------------- validation ----------------------------#
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
        "min_delta": 1e-5,
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
            "scheduler": None,
            "path": "CondOTProbPath",
            "solver": "ODESolver",
        },
        "prefix": "flow_matching/",
        "min_delta": 1e-3,
        "validation_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
        },
    },
}


# -------------------------------- Overrides -------------------------------- #

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

data_lim_1000 = {"train_dataset_size": 1000}

data_lim_2000 = {"train_dataset_size": 2000}

data_lim_3500 = {"train_dataset_size": 3500}

data_lim_5000 = {"train_dataset_size": 5000}

transformer = {
    "model": "models.temporal.TemporalTransformer",
    "model_kwargs": {
        "hidden_dim": 144,
        "hidden_dim_mult": 8,
        "depth": 8,
        "heads": 8,
        "dropout": 0.01,
        "time_embed_dim": None,
        "use_relative_pos": False,
        "recency_decay_rate": 0.0,
    },
}

"""
Configuration for Pendulum LQR (50k dataset variant).

This config contains only training/model setup. System-specific details
(state limits, preprocessing) are handled by PendulumLQRSystem.
"""
import socket
import numpy as np

from genMoPlan.utils import watch, watch_dict, get_experiments_path, process_angles
from genMoPlan.utils.systems import PendulumLQRSystem

is_arrakis = "arrakis" in socket.gethostname()
max_batch_size = int(1e6) if is_arrakis else int(266e3)


# -------------------------------- System -------------------------------- #

def get_system(config=None, use_manifold: bool = False, **kwargs):
    """
    Create a PendulumLQRSystem from this config.

    Args:
        config: Optional config dict override. If None, uses the base config.
        use_manifold: Whether to use manifold-based flow matching.
        **kwargs: Additional arguments to override system parameters.

    Returns:
        PendulumLQRSystem instance.
    """
    if config is None:
        config = base

    method_config = config.get("flow_matching", config.get("diffusion", {}))
    return PendulumLQRSystem(
        name="pendulum_lqr_50k",
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
            "n_timesteps": 5,
            "integration_method": "euler",
        },
        "final_state_directory": "final_states",
        "generated_trajectory_directory": "generated_trajectories",
        "load_ema": True,
        # System-provided inference config
        "max_path_length": _system_inference_config["max_path_length"],
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
        "early_stopping": True,
        # ---------------------------- validation ----------------------------#
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

# Non-manifold flow matching override (uses Euclidean space)
non_manifold = {
    "use_manifold": False,
    "method_kwargs": {
        "scheduler": "CondOTScheduler",
        "path": "AffineProbPath",
        "solver": "ODESolver",
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
                "mins": [-np.pi, -2 * np.pi],
                "maxs": [np.pi, 2 * np.pi],
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

dit_test = {
    "model": "models.temporal.TemporalDiffusionTransformer",
    "model_kwargs": {
        "hidden_dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "feedforward_dim": None,
        "dropout": 0.0,
        "time_embed_dim": None,
        "global_query_embed_dim": None,
        "local_query_embed_dim": None,
        "use_positional_encoding": True,
    },
    "lr_scheduler_warmup_steps": 2000,
    "learning_rate": 2.5e-4,
    "lr_scheduler_min_lr": 2e-5,
    "ema_decay": 0.999,
    "useAdamW": True,
    "optimizer_kwargs": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.02,
    },
    "clip_grad_norm": 1.0,
    "val_batch_size": int(1e4),
}

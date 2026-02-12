"""
Configuration for 1D Double Integrator with Bang-Bang control.

This config contains only training/model setup. System-specific details
(state limits, preprocessing) are handled by DoubleIntegrator1DSystem.
"""
import socket

from genMoPlan.utils import watch, watch_dict, get_experiments_path
from genMoPlan.systems import DoubleIntegrator1DSystem

is_arrakis = "arrakis" in socket.gethostname()
max_batch_size = int(1e6) if is_arrakis else int(266e3)


# -------------------------------- System -------------------------------- #

def get_system(config=None, dataset: str = None, **kwargs):
    """
    Create a DoubleIntegrator1DSystem from this config.

    Args:
        config: Optional config dict override. If None, uses the base config.
        dataset: Name of the dataset (required for loading achieved bounds).
        **kwargs: Additional arguments to override system parameters.

    Returns:
        DoubleIntegrator1DSystem instance.
    """
    if config is None:
        config = base

    # Dataset name is required
    if dataset is None:
        dataset = "double_integrator_1d_bang_bang"  # Default to config name

    method_config = config.get("flow_matching", config.get("diffusion", {}))
    return DoubleIntegrator1DSystem(
        dataset=dataset,
        stride=kwargs.get("stride", method_config.get("stride", 1)),
        history_length=kwargs.get("history_length", method_config.get("history_length", 1)),
        horizon_length=kwargs.get("horizon_length", method_config.get("horizon_length", 31)),
        variant="bang_bang",
        **{k: v for k, v in kwargs.items() if k not in ["stride", "history_length", "horizon_length", "dataset"]},
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
        "n_runs": 100,
        "batch_size": max_batch_size,
        "outcome_prob_threshold": 0.8,
        "max_path_length": 200,
        "flow_matching": {
            "n_timesteps": 5,
            "integration_method": "euler",
        },
        "final_state_directory": "final_states",
        "generated_trajectory_directory": "generated_trajectories",
    },
    "base": {
        "action_indices": None,
        "loss_type": "l2",
        "clip_denoised": False,
        "has_local_query": False,
        "has_global_query": False,
        # -------------------------------- dataset --------------------------------#
        "loader": "datasets.TrajectoryDataset",
        "shuffled_indices_fname": "shuffled_indices.txt",
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
        "min_num_steps_per_epoch": 1e4,
        "save_freq": 20,  # epochs
        "log_freq": 1e3,  # steps
        "batch_size": 64,
        "num_workers": 4,
        "learning_rate": 2e-4,
        "useAdamW": False,
        "optimizer_kwargs": {},
        "clip_grad_norm": None,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.995,
        "save_parallel": False,
        "device": "cuda",
        "seed": 42,
        # ---------------------------- validation ----------------------------#
        "val_dataset_size": 40,
        "val_batch_size": 2048,
        "val_seed": 42,
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

data_lim_10 = {"train_dataset_size": 10}
data_lim_25 = {"train_dataset_size": 25}
data_lim_50 = {"train_dataset_size": 50}
data_lim_100 = {"train_dataset_size": 100}
data_lim_500 = {"train_dataset_size": 500}
data_lim_1000 = {"train_dataset_size": 1000}
data_lim_2000 = {"train_dataset_size": 2000}
data_lim_3500 = {"train_dataset_size": 3500}
data_lim_5000 = {"train_dataset_size": 5000}

# Manifold-based flow matching (uses geodesic paths)
manifold = {
    "use_manifold": True,
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
    "min_num_steps_per_epoch": 3e3,
    "save_freq": 20,  # epochs
    "log_freq": 1e3,  # steps
    "batch_size": 64,
    "num_workers": 4,
    "learning_rate": 2e-4,
    "gradient_accumulate_every": 1,
    "ema_decay": 0.995,
    "save_parallel": False,
    "device": "cuda",
    "seed": 42,
    "min_delta": int(1e-5),
    "patience": 7,
    "early_stopping": True,
    "adaptive_training_kwargs": {
        "n_runs": 10,
        "num_inference_steps": 7,
        "sampling_batch_size": max_batch_size,
        "conditional_sample_kwargs": {
            "n_timesteps": 5,
            "integration_method": "euler",
        },
        "post_process_fns": [],
        "post_process_fn_kwargs": {},
        "combiner": "adaptive_training.ConcatCombiner",
        "combiner_kwargs": {},
        "animate_plots": True,
        "uncertainty_kwargs": {
            "inference_normalization_params": {
                "mins": [-1.01, -1.01],
                "maxs": [1.01, 1.01],
            },
        },
        "sampler": "adaptive_training.WeightedDiscreteSampler",
        "sampler_kwargs": {},
        "init_size": 100,
        "step_size": 50,
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
    "num_epochs": 1,
    "min_num_steps_per_epoch": 10,
    "adaptive_training_kwargs": {
        "uncertainty_kwargs": {
            "n_runs": 2,
        },
        "max_iters": 2,
    },
}

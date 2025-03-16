import numpy as np

from mg_diffuse.utils import watch, handle_angle_wraparound, augment_unwrapped_state_data, convert_angles_to_signed_range

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("use_horizon_padding", "H_PAD"),
    ("use_history_padding", "HIST_PAD"),
    ("predict_epsilon", "EPS"),
    ("attention", "ATN"),
    ("loss_discount", "LD"),
]

logbase = "experiments"

base = {
    "flow_matching": {
        ## model
        "model": "models.TemporalUnet",
        "flow_matching": "models.FlowMatching",
        "horizon_length": 13,
        "history_length": 3,
        "horizon_stride": 20,
        "history_stride": 20,
        "action_weight": 1,
        "action_indices": [4],
        "loss_type": "l2",
        "loss_weights": None,
        "loss_discount": 1,
        "model_kwargs": {
            "dim": 32,
            "dim_mults": (1, 4, 8, 16),
            "attention": True,
        },
        "clip_denoised": False,
        "observation_dim": 5,
        ## dataset
        "loader": "datasets.TrajectoryDataset",
        "use_plan": True,
        "dt": 0.002,
        "trajectory_normalizer": "LimitsNormalizer",
        "normalizer_params": {
            "trajectory": {
                "mins": [-np.pi, -np.pi, -30.0, -30.0, -6.0],
                "maxs": [np.pi, np.pi, 30.0, 30.0, 6.0],
            },
        },
        "trajectory_preprocess_fns": [
            convert_angles_to_signed_range, 
            handle_angle_wraparound,
            augment_unwrapped_state_data,
        ],
        "preprocess_kwargs": {
            "angle_indices": [0, 1],
        },
        "use_horizon_padding": False,
        "use_history_padding": False,
        "train_set_limit": None,
        ## serialization
        "logbase": logbase,
        "prefix": "flow_matching_trajectory/",
        "exp_name": watch(args_to_watch),
        ## training
        "n_train_steps": 1e6,
        "n_steps_per_epoch": 10000,
        "save_freq": 1e5,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": None,
    }
}

# ------------------------ overrides ------------------------#

no_plan = {
    "flow_matching": {
        "use_plan": False,
        "observation_dim": 4,
        "action_indices": None,
        "normalizer_params": {
            "trajectory": {
                "mins": [-np.pi, -np.pi, -30.0, -30.0],
                "maxs": [np.pi, np.pi, 30.0, 30.0],
            },
        },
    }
}

transformer = {
    "flow_matching": {
        "model": "models.TemporalTransformer",
        "model_kwargs": {
            "dim": 128,
            "depth": 4,
            "heads": 4,
            "dim_head": 32,
            "ff_mult": 4,
        },
    }
}

transformer_no_plan = {
    "flow_matching": {
        "model": "models.TemporalTransformer",
        "model_kwargs": {
            "dim": 128,
            "depth": 4,
            "heads": 4,
            "dim_head": 32,
            "ff_mult": 4,
        },
        "use_plan": False,
        "observation_dim": 4,
        "action_indices": None,
        "normalizer_params": {
        "trajectory": {
            "mins": [-np.pi, -np.pi, -30.0, -30.0],
                "maxs": [np.pi, np.pi, 30.0, 30.0],
            },
        },
    }
}

transformer_large = {
    "flow_matching": {
        "model": "models.TemporalTransformer",
        "model_kwargs": {
            "dim": 320,
            "depth": 8,
            "heads": 10,
            "dim_head": 64,
            "ff_mult": 6,
        },
        "loss_discount": 0.99,
    }
}
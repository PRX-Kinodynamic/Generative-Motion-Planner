import numpy as np

from mg_diffuse.utils import watch, handle_angle_wraparound, augment_unwrapped_state_data, convert_angles_to_signed_range

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("use_padding", "PAD"),
    ("predict_epsilon", "EPS"),
    ("attention", "ATN"),
    ("loss_discount", "LD"),
]

logbase = "experiments"

base = {
    "flow_matching": {
        ## model
        "model": "models.FlowMatching",
        "horizon": 50,
        "history_length": 50,
        "action_weight": 10,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (1, 2, 4, 8),
        "attention": False,
        "clip_denoised": False,
        "observation_dim": 2,
        ## dataset
        "loader": "datasets.PlanDataset",
        "dt": 0.002,
        "trajectory_normalizer": "LimitsNormalizer",
        "plan_normalizer": "LimitsNormalizer",
        "normalizer_params": {
            "trajectory": {
                "mins": [-np.pi, -np.pi, -30.0, -30.0],
                "maxs": [np.pi, np.pi, 30.0, 30.0],
            },
            "plan": {
                "mins": [-6.0],
                "maxs": [6.0],
            },
        },
        "trajectory_preprocess_fns": [
            convert_angles_to_signed_range, 
            handle_angle_wraparound,
            augment_unwrapped_state_data,
        ],
        "plan_preprocess_fns": [],
        "preprocess_kwargs": {
            "angle_indices": [0, 1],
        },
        "use_history_padding": False,
        "train_set_limit": None,
        ## serialization
        "logbase": logbase,
        "prefix": "flow_matching/",
        "exp_name": watch(args_to_watch),
        ## training
        "loss_type": "l2",
        "n_train_steps": 5e5,
        "n_steps_per_epoch": 5000,
        "save_freq": 1e5,
        "sample_freq": 1e4,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": None,
        ## visualization
        "sampling_limits": (-1, 1),
        "granularity": 0.01,
    }
}

# ------------------------ overrides ------------------------#


from mg_diffuse.utils import watch, handle_angle_wraparound

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
    ## value kwargs
    ("discount", "d"),
]

logbase = "experiments"

base = {
    "roa_estimation": {
        "attractors": {
            (-2.1, 0): 0,
            (2.1, 0): 0,
            (0, 0): 1,
        },
        "invalid_label": -1,
        "attractor_threshold": 0.05,
        "n_runs": 20,
        "batch_size": 200000,
        "attractor_probability_upper_threshold": 0.8,
    },

    "diffusion": {
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 32,
        "n_diffusion_steps": 20,
        "action_weight": 10,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (1, 2, 4, 8),
        "attention": False,
        "clip_denoised": False,
        "observation_dim": 2,
        ## dataset
        "loader": "datasets.TrajectoryDataset",
        "normalizer": "LimitsNormalizer",
        "preprocess_fns": [handle_angle_wraparound],
        "preprocess_kwargs": {
            "augment_new_state_data": False,
        },
        "use_padding": True,
        "max_path_length": 502,
        "train_set_limit": None,
        ## serialization
        "logbase": logbase,
        "prefix": "diffusion/",
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

fewer_steps = {
    "diffusion": {
        "n_diffusion_steps": 5,
    }
}

one_step = {
    "diffusion": {
        "n_diffusion_steps": 1,
    }
}

long_horizon = {
    "diffusion": {
        "horizon": 80,
    }
}

longer_horizon = {
    "diffusion": {
        "horizon": 160,
    }
}

data_lim_100 = {
    "diffusion": {
        "train_set_limit": 100
    }
}

data_lim_500 = {
    "diffusion": {
        "train_set_limit": 500
    }
}

data_lim_1000 = {
    "diffusion": {
        "train_set_limit": 1000
    }
}

no_preprocess = {
    "diffusion": {
        "preprocess_fns": []
    }
}

no_augment = {
    "diffusion": {
        "preprocess_kwargs": {
            "augment_new_state_data": False,
        }
    }
}



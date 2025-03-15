from mg_diffuse.utils import watch, handle_angle_wraparound
import numpy as np

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("method_steps", "T"),
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

    "manifold": {
        "sphere_dim":0,
        "torus_dim":2,
        "euclidean_dim":3
    },

    "method": {
        ## model
        # "type": "flowmatching",
        "model": "models.TemporalUnet",
        "flowmatching": "models.FlowMatching",
        "method_type":"models.FlowMatching",
        "horizon": 16,
        "method_steps": 20,
        "action_weight": 10,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (1, 2, 4, 8),
        "attention": True,
        "clip_denoised": False,
        "observation_dim": 5,
        ## dataset
        "loader": "datasets.TrajectoryDataset",
        "stride": 16,
        "dt": 0.002,
        "normalizer": "WrapManifold",
        "plan_normalizer": "LimitsNormalizer",
        "use_plans": True,
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
        "trajectory_preprocess_fns": [],
        "plan_preprocess_fns": [],
        "preprocess_kwargs": {
            "angle_indices": [0, 1],
        },
        "use_history_padding": False,
        "train_set_limit": None,
        "use_padding" : False,
        "max_path_length":2000,
        ## serialization
        "logbase": logbase,
        "prefix": "flowmatching/",
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
    "flowmatching": {
        "method_steps": 5,
    }
}

one_step = {
    "flowmatching": {
        "method_steps": 1,
    }
}

long_horizon = {
    "flowmatching": {
        "horizon": 80,
    }
}

longer_horizon = {
    "flowmatching": {
        "horizon": 160,
    }
}

data_lim_100 = {
    "flowmatching": {
        "train_set_limit": 100
    }
}

data_lim_500 = {
    "flowmatching": {
        "train_set_limit": 500
    }
}

data_lim_1000 = {
    "flowmatching": {
        "train_set_limit": 1000
    }
}

no_preprocess = {
    "flowmatching": {
        "preprocess_fns": []
    }
}

no_augment = {
    "flowmatching": {
        "preprocess_kwargs": {
            "augment_new_state_data": False,
        }
    }
}



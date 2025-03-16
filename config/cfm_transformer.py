import numpy as np

from mg_diffuse.utils import watch
from mg_diffuse.utils.data_preprocessing import convert_angles_to_signed_range, handle_angle_wraparound, augment_unwrapped_state_data

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("horizon_stride", "HS"),
    ("history_length", "HL"),
    ("history_stride", "HS"),
    ("dim", "DIM"),
    ("depth", "DEPTH"),
    ("heads", "HEADS"),
    ("cross_attention", "CRSATN"),
    ("dropout", "DROP"),
]

logbase = "experiments"

base = {
    "flow_matching": {
        ## model
        "model": "models.transformer.PlanTransformerFlowMatching",
        "horizon": 50,
        "horizon_stride": 1,
        "history_length": 50,
        "history_stride": 1,
        "n_timesteps": 20,
        "step_size": 0.05,
        "integration_method": "midpoint",
        "dim": 256,
        "depth": 6,
        "heads": 8,
        "dim_feedforward": 512,
        "dropout": 0,
        "cross_attention": True,
        ## dataset
        "loader": "datasets.plan.PlanDataset",
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
        ],
        "plan_preprocess_fns": [],
        "preprocess_kwargs": {
            "angle_indices": [0, 1],
        },
        "use_history_padding": True,
        "train_set_limit": None,
        ## serialization
        "logbase": logbase,
        "prefix": "transformer_flow_matching/",
        "exp_name": watch(args_to_watch),
        ## training
        "loss_type": "l2",
        "n_train_steps": 100000,
        "n_steps_per_epoch": 1000,
        "save_freq": 10000,
        "sample_freq": 5000,
        "batch_size": 64,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.995,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": None,
        "patience": 10,
        "min_delta": 1e-4,
        ## visualization
        "sampling_limits": (-1, 1),
        "granularity": 0.01,
    }
}

# ------------------------ overrides ------------------------#


small_fast_model = {
    "flow_matching": {
        ## smaller model architecture
        "dim": 128,
        "depth": 3,
        "heads": 4,
        "dim_feedforward": 256,
        "dropout": 0,
        
        ## reduced sequence lengths
        "horizon": 30,
        "history_length": 30,
        
        ## lighter training configuration
        "batch_size": 64,
        "learning_rate": 3e-4,
        "n_train_steps": 50000,
        "n_steps_per_epoch": 500,
        "save_freq": 5000,
        "sample_freq": 2500,
        
        ## experiment naming
        "prefix": "transformer_flow_matching_small/",
    }
}

stride_30 = {
    "flow_matching": {
        ## even smaller model architecture (since processing fewer timesteps)
        "dim": 64,
        "depth": 4,
        "heads": 4,
        "dim_feedforward": 128,
        "dropout": 0,
        
        ## reduced sequence lengths with increased stride
        "horizon": 16,
        "horizon_stride": 30,
        "history_length": 16,
        "history_stride": 30,
        
        ## lighter training configuration
        "batch_size": 128,
        "learning_rate": 5e-4,
        "n_train_steps": 50000,
        "n_steps_per_epoch": 500,
        "save_freq": 3000,
        "sample_freq": 1500,
        
        ## experiment naming
        "prefix": "transformer_flow_matching_strider_30/",
    }
}

stride_45 = {
    "flow_matching": {
        ## ultra-compact model architecture for very sparse timesteps
        "dim": 64,
        "depth": 4,
        "heads": 4,
        "dim_feedforward": 64,
        "dropout": 0,
        
        ## minimal sequence lengths with large stride
        "horizon": 16,
        "horizon_stride": 45,
        "history_length": 16,
        "history_stride": 45,
        
        ## efficient training configuration
        "batch_size": 128,
        "learning_rate": 8e-4,
        "n_train_steps": 50000,
        "n_steps_per_epoch": 500,
        "save_freq": 2000,
        "sample_freq": 1000,
        
        ## experiment naming
        "prefix": "transformer_flow_matching_strider_45/",
    }
}


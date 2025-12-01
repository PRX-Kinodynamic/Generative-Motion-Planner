import socket
from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product, Sphere
import numpy as np

from genMoPlan.utils import watch, watch_dict, shift_to_zero_center_angles, handle_angle_wraparound, augment_unwrapped_state_data, process_angles, get_experiments_path

is_arrakis = 'arrakis' in socket.gethostname()

max_batch_size = int(1e5) if is_arrakis else int(4e3)

mins = [
    -0.859485, -1.375191, -0.688990, -0.525767, -0.632861, -1.994933, -2.905200, -0.540604, -1.030145, -0.544243, -0.614817, -1.993659, -2.883713, -0.487868, -1.017001, -1.579886, -1.602320, -1.720383, -1.183831, -1.170331, -1.739313, 0.051704, -0.455080, -0.168303, -0.557510, -0.858317, -0.986132, -1.281937, -0.446948, -0.786912, -0.556428, -0.895774, -1.038971, -1.279955, 
    -1.000000, -0.999984, -0.896724, 
    -1.914248, -1.915843, -3.327530, -2.658107, -2.528388, -5.331130, -11.301671, -13.006869, -17.815111, -14.461623, -14.078833, -12.893085, -12.061430, -12.384354, -12.161468, -17.793238, -26.302641, -20.862368, -10.522175, -13.214445, -13.203420, -20.225025, -19.451426, -21.217875, -24.788021, -18.834494, -21.369850, -22.102814, -20.644373, -21.245199,
]
maxs = [
    0.838374, 0.609621, 0.725149, 0.164440, 0.517798, 0.522608, 0.185058, 0.921429, 0.983391, 0.163125, 0.545136, 0.491036, 0.209521, 0.921906, 1.020330, 1.199831, 1.143854, 0.985109, 1.646968, 1.620824, 0.964722, 1.494286, 0.617372, 0.787087, 0.556917, 1.055067, 1.020654, 0.523082, 0.618061, 0.187432, 0.555102, 1.036542, 0.980308, 0.523863, 
    1.000000, 1.000000, 1.000000, 
    1.900168, 1.837195, 1.918130, 2.789604, 2.940170, 2.997349, 8.091124, 12.754802, 14.861890, 11.852520, 14.249229, 11.141454, 11.107437, 12.403696, 14.443352, 25.315464, 19.668034, 18.593166, 11.033415, 12.674791, 17.079679, 24.056908, 22.487865,
    21.304333, 23.978119, 19.407972, 24.520531, 25.825670, 21.328411, 24.432425,
]

manifold_mins = mins.copy()
manifold_maxs = maxs.copy()

manifold_mins[-3:] = [None] * 3
manifold_maxs[-3:] = [None] * 3

manifold = Product(
    input_dim=67,
    manifolds=[
        (Euclidean(), 34),
        (Sphere(), 3)
        (Euclidean(), 30)
    ],
)

angle_indices = []

def read_trajectory(sequence_path):
    with open(sequence_path, "r") as f:
        lines = f.readlines()

    trajectory = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line == "":
            if i < len(lines) - 1:
                raise ValueError(f"[ config/humanoid_get_up ] Empty line found at {sequence_path} at line {i}")
            else:
                break

        state = line.split(',')

        state = [s for s in state if s != ""]

        if len(state) < 67:
            raise ValueError(f"[ config/humanoid_get_up ] Trajectory at {sequence_path} has less than 67 states at line {i}")

        state = state[:67]

        state = np.array([float(s) for s in state])

        trajectory.append(state)

    return np.array(trajectory, dtype=np.float32)

exp_args_to_watch = [
    ("history_length", "HILEN"),
    ("horizon_length", "HOLEN"),
    ("use_history_padding", "HIPAD"),
    ("use_horizon_padding", "HOPAD"),
    ("stride", "STRD"),
]

results_args_to_watch = [
    ("n_runs", "NRUN"),
    ("attractor_dist_threshold", "ADTH"),
    ("attractor_prob_threshold", "APTH"),
]

logbase = get_experiments_path()


base = {
    "inference": {
        "results_name": watch_dict(results_args_to_watch),
        # "attractors": {
        #     (-2.1, 0): 0,
        #     (2.1, 0): 0,
        #     (0, 0): 1,
        # },
        "invalid_label": -1,
        "n_runs": 20,
        "batch_size": max_batch_size,
        "attractor_dist_threshold": 0.075,
        "attractor_prob_threshold": 0.6,
        "max_path_length": 745,
        "flow_matching": {
            "n_timesteps": 5,
            "integration_method": "euler",
        },
        "post_process_fns": [
            process_angles,
        ],
        "post_process_fn_kwargs": {
            "angle_indices": [0],
        },
        "final_state_directory": "final_states",
        "generated_trajectory_directory": "generated_trajectories",
        "manifold_unwrap_fns": [shift_to_zero_center_angles],
        "manifold_unwrap_kwargs": {
            "angle_indices": [0],
        },
        "load_ema": True,
    },

    "base": {
        "action_indices": None,
        "angle_indices": angle_indices,
        "loss_type": "l2",
        "clip_denoised": False,
        "observation_dim": 67,
        "has_local_query": False,
        "has_global_query": False,

        #-------------------------------- dataset --------------------------------#
        "loader": "datasets.TrajectoryDataset",
        "read_trajectory_fn": read_trajectory,
        "trajectory_normalizer": "LimitsNormalizer",
        "plan_normalizer": None,
        "normalizer_params": {
            "trajectory": {
                "mins": mins,
                "maxs": maxs,
            },
            "plan": None,
        },
        "plan_preprocess_fns": None,    
        "trajectory_preprocess_fns": [
            handle_angle_wraparound,
            augment_unwrapped_state_data,
        ],
        "preprocess_kwargs": {
            "trajectory": {
                "angle_indices": angle_indices,
            },
            "plan": None,
        },
        "use_history_padding": False,
        "use_horizon_padding": True,
        "use_history_mask": False,
        "use_plan": False,
        "train_dataset_size": None,
        "is_history_conditioned": True,

        #---------------------------- serialization ----------------------------#
        "logbase": logbase,
        "exp_name": watch(exp_args_to_watch),

        "dataset_kwargs": {
            "cost_mul_threshold": 1.0,
        },

        #---------------------------- training ----------------------------#
        "num_epochs": 100,
        "min_num_steps_per_epoch": 0,
        "save_freq": 20, # epochs
        "log_freq": 1e2, # steps
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

        #---------------------------- early stopping-------------------------#
        "patience": 10,
        "warmup_epochs": 5,
        "early_stopping": False,

        #---------------------------- validation ----------------------------#
        "val_dataset_size": 100,
        "val_batch_size": max_batch_size,
        "val_seed": 42,

        #-------------------------------evaluation--------------------------#
        "perform_final_state_evaluation": True,
        "eval_freq": 10, # epochs
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
        "manifold": None,
    },

    "flow_matching": {
        "method_name": "flow_matching",
        "method": "models.generative.FlowMatching",
        "manifold": manifold,
        "manifold_unwrap_fns": [],
        "manifold_unwrap_kwargs": {
            "angle_indices": angle_indices,
        },
        "trajectory_preprocess_fns": [],
        "preprocess_kwargs": {},
        "normalizer_params": {
            "trajectory": {
                "mins": manifold_mins,
                "maxs": manifold_maxs,
            },
            "plan": None,
        },
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
    }
}

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

larger_hidden_dim = {
    "model_kwargs": {
        "base_hidden_dim": 64,
    },
}

lr_test = {
    "learning_rate": 2e-4,            # peak (was 1e-4)
    "lr_scheduler_warmup_steps": 3000, # ~9.4 epochs @ 319 steps/epoch
    "lr_scheduler_min_lr": 5e-5,      # raise floor (was 2e-5)
    "ema_decay": 0.999,               # slightly stronger EMA for the higher LR
    "gradient_accumulate_every": 2,   # optional: 2x effective batch if you want steadier grads
}

dit_test = {
    "model": "models.temporal.TemporalDiffusionTransformer",
    "model_kwargs": {
        "hidden_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "feedforward_dim": None,
        "dropout": 0.0,
        "time_embed_dim": None,
        "global_query_embed_dim": None,
        "local_query_embed_dim": None,
        "use_positional_encoding": True,
    },

    "lr_scheduler_warmup_steps": 1000,
    "learning_rate": 2e-4,
    "lr_scheduler_min_lr": 2e-5,
    "ema_decay": 0.999,
    "useAdamW": True,
    "optimizer_kwargs": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.01,
    },
    "clip_grad_norm": 1.0,

    "val_batch_size": int(6e4) if is_arrakis else int(1e4),
}

dit_dropout = {
    "model_kwargs": {
        "dropout": 0.1,
    },
}

dit_lr_test_a = {
    "learning_rate": 1e-4,
    "lr_scheduler_warmup_steps": 1000,
    "optimizer_kwargs": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.01,
    },

}

dit_lr_test_b = {
    "learning_rate": 1e-4,
    "lr_scheduler_warmup_steps": 500,
    "optimizer_kwargs": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.01,
    },
}

dit_lr_test_c = {
    "learning_rate": 1.5e-4,
    "lr_scheduler_warmup_steps": 1000,
    "optimizer_kwargs": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.01,
    },
}

dit_lr_test_d = {
    "learning_rate": 2e-4,
    "lr_scheduler_warmup_steps": 1000,
    "optimizer_kwargs": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.01,
    },
}

dit_lr_test_e = {
    "learning_rate": 1e-4,
    "lr_scheduler_warmup_steps": 1000,
    "optimizer_kwargs": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.005,
    },
}

dit_lr_test_f = {
    "learning_rate": 1e-4,
    "lr_scheduler_warmup_steps": 1000,
    "optimizer_kwargs": {
        "betas": (0.9, 0.95),
        "weight_decay": 0.02,
    },
}

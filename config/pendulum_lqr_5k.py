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
    "diffusion": {
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 32,
        "n_diffusion_steps": 20,
        "action_weight": 10,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": True,
        "dim_mults": (1, 2, 4, 8),
        "attention": False,
        "clip_denoised": False,
        "observation_dim": 2,
        ## dataset
        "loader": "datasets.TrajectoryDataset",
        "normalizer": "LimitsNormalizer",
        "preprocess_fns": [handle_angle_wraparound],
        "use_padding": True,
        "max_path_length": 502,
        ## serialization
        "logbase": logbase,
        "prefix": "diffusion/",
        "exp_name": watch(args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "l2",
        "n_train_steps": 1e6,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_freq": 20000,
        "sample_freq": 20000,
        "n_saves": 5,
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

# pendulum_lqr_5k = {
#     "diffusion": {
#         "max_path_length": 502,
#     }
# }

from os import path

import torch

from mg_diffuse.utils import JSONArgs, import_class


def load_model_args(experiments_path):
    model_args_path = path.join(experiments_path, "args.json")
    return JSONArgs(model_args_path)


def load_model(experiments_path, model_state_name, verbose=False):
    model_args = load_model_args(experiments_path)
    model_path = path.join(experiments_path, model_state_name)

    if verbose:
        print(f"[ scripts/visualize_trajectories ] Loading model from {model_path}")

    model_state_dict = torch.load(model_path, weights_only=False)
    diff_model_state = model_state_dict["model"]

    model_class = import_class(model_args.model)
    diffusion_class = import_class(model_args.diffusion)

    model = model_class(
        horizon=model_args.horizon,
        transition_dim=model_args.observation_dim,
        cond_dim=model_args.observation_dim,
        dim_mults=model_args.dim_mults,
        attention=model_args.attention,
    ).to(model_args.device)

    diffusion = diffusion_class(
        model=model,
        horizon=model_args.horizon,
        observation_dim=model_args.observation_dim,
        n_timesteps=model_args.n_diffusion_steps,
        loss_type=model_args.loss_type,
        clip_denoised=model_args.clip_denoised,
        predict_epsilon=model_args.predict_epsilon,
        ## loss weighting
        loss_weights=model_args.loss_weights,
        loss_discount=model_args.loss_discount,
    ).to(model_args.device)

    # Load model state dict
    diffusion.load_state_dict(diff_model_state)

    return diffusion, model_args

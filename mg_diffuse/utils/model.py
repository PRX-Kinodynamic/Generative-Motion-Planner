from os import path

import torch

from mg_diffuse.utils import JSONArgs, import_class

from flow_matching.utils.manifolds import Product


def load_model_args(experiments_path):
    model_args_path = path.join(experiments_path, "args.json")
    return JSONArgs(model_args_path)


def load_model(experiments_path, model_state_name, verbose=False):
    model_args = load_model_args(experiments_path)
    model_path = path.join(experiments_path, model_state_name)

    if verbose:
        print(f"[ scripts/visualize_trajectories ] Loading model from {model_path}")

    model_state_dict = torch.load(model_path, weights_only=False)
    method_model_state = model_state_dict["model"]

    model_class = import_class(model_args.model)
    method_class = import_class(model_args.method_type)

    # sphere and torus have two features for dimension (cos, sin)
    features_dim = 2*model_args.sphere_dim + 2*model_args.torus_dim + model_args.euclidean_dim

    model = model_class(
        horizon=model_args.horizon,
        transition_dim=features_dim,
        cond_dim=model_args.observation_dim,
        dim_mults=model_args.dim_mults,
        attention=model_args.attention,
    ).to(model_args.device)

    method = method_class(
        model=model,
        horizon=model_args.horizon,
        observation_dim=model_args.observation_dim,
        n_timesteps=model_args.method_steps,
        loss_type=model_args.loss_type,
        clip_denoised=model_args.clip_denoised,
        predict_epsilon=model_args.predict_epsilon,
        ## loss weighting
        loss_weights=model_args.loss_weights,
        loss_discount=model_args.loss_discount,
        manifold=Product(model_args.sphere_dim, model_args.torus_dim, model_args.euclidean_dim)
    ).to(model_args.device)

    # Load model state dict
    method.load_state_dict(method_model_state)

    return method, model_args



    # if "model_args.diffusion" in locals():
    #     diffusion_class = import_class(model_args.diffusion)
    #     diffusion = diffusion_class(
    #         model=model,
    #         horizon=model_args.horizon,
    #         observation_dim=model_args.observation_dim,
    #         n_timesteps=model_args.n_diffusion_steps,
    #         loss_type=model_args.loss_type,
    #         clip_denoised=model_args.clip_denoised,
    #         predict_epsilon=model_args.predict_epsilon,
    #         ## loss weighting
    #         loss_weights=model_args.loss_weights,
    #         loss_discount=model_args.loss_discount,
    #     ).to(model_args.device)

    #     # Load model state dict
    #     diffusion.load_state_dict(diff_model_state)
    #     method = diffusion

    # else:
    #     flowmatching_class = import_class(model_args.flowmatching)
    #     flowmatching = flowmatching_class(
    #         model=model,
    #         horizon=model_args.horizon,
    #         observation_dim=model_args.observation_dim,
    #         n_timesteps=model_args.flowmatching_steps,
    #         loss_type=model_args.loss_type,
    #         clip_denoised=model_args.clip_denoised,
    #         predict_epsilon=model_args.predict_epsilon,
    #         ## loss weighting
    #         loss_weights=model_args.loss_weights,
    #         loss_discount=model_args.loss_discount,
    #     ).to(model_args.device)

    #     # Load model state dict
    #     flowmatching.load_state_dict(diff_model_state)
    #     method = flowmatching

    # return method, model_args

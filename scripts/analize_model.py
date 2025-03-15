from os import path
import argparse
import numpy as np
from tqdm import tqdm

import mg_diffuse.utils.model
from mg_diffuse import utils
import torch
from mg_diffuse.utils.plan import load_plans, combine_plans_trajectories

from flow_matching.utils.manifolds import Product

from mg_diffuse.datasets.sequence import TrajectoryDataset
from mg_diffuse.datasets.normalization import WrapManifold


def analize_model(dataset, num_trajs, compare, show_traj_ends, exp_name, model_state_name, only_execute_next_step, path_prefix="diffusion"):
    exp_path = path.join("experiments", dataset, path_prefix, exp_name) # original with bug
    exp_path = path.join("experiments", dataset, exp_name) # solution

    model, model_args = mg_diffuse.utils.model.load_model(exp_path, model_state_name)

    manifold=Product(sphere_dim = model_args.sphere_dim, torus_dim = model_args.torus_dim, euclidean_dim = model_args.euclidean_dim)

   
    dataset = TrajectoryDataset(
        dataset=dataset,
        horizon=model_args.horizon,
        stride=model_args.stride,
        normalizer=model_args.normalizer,
        normalizer_params=model_args.normalizer_params,
        max_path_length=model_args.horizon,
        dataset_size=num_trajs,
        use_plans=model_args.use_plans,
        manifold=manifold,
    )


    ### Load normalization parameters
    # Merge trajectory and plan normalization parameters
    normalizer_params = model_args.normalizer_params
    if model_args.use_plans:
        merged_traj_plan_norm_params = {
            "mins": normalizer_params["trajectory"]["mins"] + normalizer_params["plan"]["mins"],
            "maxs": normalizer_params["trajectory"]["maxs"] + normalizer_params["plan"]["maxs"],
        }
    else: # No plans
        merged_traj_plan_norm_params = model_args.normalizer_params["trajectory"]

    normalizer = WrapManifold(manifold, params=merged_traj_plan_norm_params)
    ###

    # Initialize lists to store trajectories and conditions
    trajectories_list = []
    conditions_list = []
    # Loop over dataset and extract trajectories and conditions
    for data in tqdm(dataset):
        trajectories_list.append(data.trajectories)
        conditions_list.append(list(data.conditions.values())[0])

    # Stack tensors to create batches
    trajectories_test = torch.stack(trajectories_list)  # Shape: (batch_size, sequence_length, feature_dim)
    conditions_batch = {0: torch.stack(conditions_list)}  # Shape: (batch_size, feature_dim)

    trajectories_prediction = model.forward(
        conditions_batch, horizon=model_args.horizon, verbose=False
    ).trajectories


    trajectories_prediction = normalizer.unnormalize(trajectories_prediction.cpu().numpy())
    trajectories_test = normalizer.unnormalize(trajectories_test.cpu().numpy())


    error = np.linalg.norm(trajectories_prediction - trajectories_test, axis=(0,2)) /trajectories_prediction.shape[0]
    # /(trajectories_prediction.shape[0]*trajectories_prediction.shape[2])

    print("Average error: ", error)

    # Compute cumulative error along axis 0
    cumulative_error = np.cumsum(error, axis=0)

    print("Cumulative error:", cumulative_error)

    error = np.linalg.norm(trajectories_prediction - trajectories_test)/trajectories_prediction.shape[0]
    # /(trajectories_prediction.shape[0]*trajectories_prediction.shape[1]*trajectories_prediction.shape[2])

    print("Overall error: ", error)
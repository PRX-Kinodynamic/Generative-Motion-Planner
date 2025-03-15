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
    merged_traj_plan_norm_params = {
        "mins": normalizer_params["trajectory"]["mins"] + normalizer_params["plan"]["mins"],
        "maxs": normalizer_params["trajectory"]["maxs"] + normalizer_params["plan"]["maxs"],
    }
    normalizer = WrapManifold(manifold, params=merged_traj_plan_norm_params)
    ###

    
    trajectories_predition = []
    trajectories_test = []

    for i in tqdm(range(len(dataset))):
        batch = utils.batchify(dataset[i])
        cond = batch[1] 

        traj_pred = model.forward(
                cond, horizon=model_args.horizon, verbose=False
            ).trajectories

        # Apply unnormalization to the predicted trajectory   
        traj_pred_norm = normalizer.unnormalize(traj_pred.cpu().numpy()[0]) #traj_pred_norm.cpu().numpy()[0]
        trajectories_predition.append(traj_pred_norm)


        # Apply unnormalization to the test trajectory
        traj_test_norm = normalizer.unnormalize(batch[0].cpu().numpy()[0])
        trajectories_test.append(traj_test_norm)

        
    trajectories_prediction = np.stack(trajectories_predition)
    trajectories_test = np.array(trajectories_test)

    error = np.linalg.norm(trajectories_prediction - trajectories_test, axis=(0,2)) /trajectories_prediction.shape[0]
    # /(trajectories_prediction.shape[0]*trajectories_prediction.shape[2])


    print("Average error: ", error)



    # Compute cumulative error along axis 0
    cumulative_error = np.cumsum(error, axis=0)

    print("Cumulative error:", cumulative_error)

    error = np.linalg.norm(trajectories_prediction - trajectories_test)/trajectories_prediction.shape[0]
    # /(trajectories_prediction.shape[0]*trajectories_prediction.shape[1]*trajectories_prediction.shape[2])

    print("Overall error: ", error)
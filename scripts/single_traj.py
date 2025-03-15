import mg_diffuse.utils as utils
from analize_model import analize_model
from estimate_roa import generate_and_analyze_runs
import sys
import argparse
import mg_diffuse.utils.model
from flow_matching.utils.manifolds import Euclidean, Sphere, FlatTorus, Product
from mg_diffuse.datasets.normalization import WrapManifold

import os.path as path

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

def single_traj(dataset, exp_name, model_state_name):
    exp_path = path.join("experiments", dataset, exp_name)

    model, model_args = mg_diffuse.utils.model.load_model(exp_path, model_state_name)

    manifold=Product(sphere_dim = model_args.sphere_dim, torus_dim = model_args.torus_dim, euclidean_dim = model_args.euclidean_dim)


    ### Load normalization parameters
    # Merge trajectory and plan normalization parameters
    normalizer_params = model_args.normalizer_params
    merged_traj_plan_norm_params = {
        "mins": normalizer_params["trajectory"]["mins"] + normalizer_params["plan"]["mins"],
        "maxs": normalizer_params["trajectory"]["maxs"] + normalizer_params["plan"]["maxs"],
    }
    normalizer = WrapManifold(manifold, params=merged_traj_plan_norm_params)
    ###

    starting_point = np.array([[-6,6,30,-30,6]])
    starting_point_norm = normalizer.normalize(starting_point)
    starting_point_norm =torch.tensor(starting_point_norm)
    
    cond = {0:starting_point_norm}

    traj_pred = model.forward(
            cond, horizon=model_args.horizon, verbose=False
        ).trajectories

    traj_pred = normalizer.unnormalize(traj_pred.cpu().numpy())

    print(traj_pred[0])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Single trajectory analysis")
    parser.add_argument("--exp_name", type=str, required=False, help="Experiment name")
    parser.add_argument("--dataset", type=str, required=False, help="dataset name")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    
    
    args = parser.parse_args()


    single_traj(
        dataset=args.dataset,
        exp_name=args.exp_name,
        model_state_name='best.pt',
    )

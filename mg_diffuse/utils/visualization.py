import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from os import path, cpu_count

from .json_args import JSONArgs
from .config import import_class
from mg_diffuse.datasets.normalization import LimitsNormalizer
from mg_diffuse.datasets.sequence import load_trajectories


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


def plot_trajectory(trajectory):
    for i in range(len(trajectory)):
        plt.scatter(
            trajectory[i, 0],
            trajectory[i, 1],
            s=0.1,
            color="black",
            alpha=0.9,
            marker=".",
        )


def generate_trajectories(
    model, model_args, start_states, only_execute_next_step, verbose=False
):
    """
    Generate a trajectory from the model given the start states.

    Args:
        model: The model to generate the trajectory from.
        model_args: The arguments used to generate the model.
        start_states: The initial states to start the trajectory from.
            batch_size x observation_dim
        only_execute_next_step: If True, only execute the next step of the trajectory. (like MPC)
        verbose: If True, print progress.
    """
    if verbose:
        print("[ scripts/visualize_trajectories ] Generating trajectories")

    max_path_length = model_args.max_path_length
    batch_size = len(start_states)

    current_states = torch.tensor(start_states, dtype=torch.float32).to(
        model_args.device
    )
    current_idx = 1
    next_path_lengths = model_args.horizon - 1 if not only_execute_next_step else 1

    trajectories = np.zeros((batch_size, max_path_length, model_args.observation_dim))
    trajectories[:, 0] = np.array(start_states)

    with tqdm(total=max_path_length) as pbar:
        while current_idx < max_path_length:
            conditions = {0: current_states}

            # Forward pass to get the next states
            next_trajs = model.forward(
                conditions, horizon=model_args.horizon, verbose=False
            ).trajectories

            # Check what is the size of trajectory required to reach max_path_length
            slice_path_length = min(next_path_lengths, max_path_length - current_idx)

            # Removing the start state and taking only the required path lengths
            next_trajs = next_trajs[:, 1 : 1 + slice_path_length]

            # Adding the next states to the trajectory
            trajectories[:, current_idx : current_idx + slice_path_length] = (
                next_trajs.cpu().numpy()
            )

            current_states = next_trajs[:, -1]
            current_idx += next_path_lengths

            pbar.update(slice_path_length)

    return trajectories


def process_trajectories(trajectories, model_args):
    """
    Process the trajectories to make them more interpretable.
    """
    params = {
        "maxs": [
            6.283153,
            6.19742,
        ],
        "mins": [
            -6.2831573,
            -6.28318,
        ],
    }

    normalizer = LimitsNormalizer(params=params)

    processed_trajectories = []

    for trajectory in trajectories:
        trajectory = normalizer.unnormalize(trajectory)
        # If value of trajectory[0] is greater than pi, then subtract 2pi from the trajectory
        trajectory[trajectory[:, 0] > np.pi, 0] -= 2 * np.pi
        trajectory[trajectory[:, 0] < -np.pi, 0] += 2 * np.pi

        processed_trajectories.append(trajectory)

    trajectories = np.array(processed_trajectories)

    return trajectories


def save_trajectories_image(trajectories, image_path, verbose=False):
    """
    Visualize the trajectories generated by the model.
    """

    if verbose:
        print("[ utils/visualization ] Visualizing trajectories...")

    # if not parallel:
    for idx in tqdm(range(trajectories.shape[0])):
        # plot_trajectory(trajectories[idx])
        trajectory = trajectories[idx]
        plt.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            s=0.1,
            color="black",
            alpha=0.9,
            marker=".",
        )
        # for i in range(len(trajectory)):
        #     plt.scatter(
        #         trajectory[i, 0],
        #         trajectory[i, 1],
        #         s=0.1,
        #         color="black",
        #         alpha=0.9,
        #         marker=".",
        #     )

    # else:
    #     from multiprocessing import Pool
    #
    #     with Pool(cpu_count()-1) as pool:
    #         # Start pool and visualize progress with tqdm
    #         list(tqdm(pool.imap(plot_trajectory, trajectories), total=trajectories.shape[0]))

    plt.savefig(image_path)

    if verbose:
        print(f"[ utils/visualization ] Trajectories saved at {image_path}")


def get_fnames_to_load(dataset_path, num_trajs):
    indices_file = path.join(dataset_path, "viz_shuffled_indices.txt")

    if path.exists(indices_file):
        with open(indices_file, "r") as f:
            fnames = f.readlines()
            fnames = [f.strip() for f in fnames]
            fnames = fnames[:num_trajs]

    else:
        raise ValueError(f"File {indices_file} not found")

    return fnames


def load_test_trajectories(dataset, num_trajs):
    dataset_path = path.join("data_trajectories", dataset)

    fnames = get_fnames_to_load(dataset_path, num_trajs)

    return load_trajectories(dataset, parallel=True, fnames=fnames)

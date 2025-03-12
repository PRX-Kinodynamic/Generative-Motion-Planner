from tqdm import tqdm
import matplotlib.pyplot as plt
from os import cpu_count, path, listdir
from random import shuffle

import torch
import numpy as np

from mg_diffuse.datasets.normalization import WrapManifold
from flow_matching.utils.manifolds import Product


def generate_trajectory_batch(start_states, model, model_args, only_execute_next_step=False):
    batch_size = len(start_states)
    max_path_length = model_args.max_path_length

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
            next_trajs = next_trajs[:, 1: 1 + slice_path_length]

            # Adding the next states to the trajectory
            trajectories[:, current_idx: current_idx + slice_path_length] = (
                next_trajs.cpu().numpy()
            )

            current_states = next_trajs[:, -1]
            current_idx += next_path_lengths

            pbar.update(slice_path_length+1)

    return trajectories

def generate_trajectories(
    model, model_args, unnormalized_start_states, only_execute_next_step, return_normalize=False, verbose=False, batch_size=5000
):
    """
    Generate a trajectory from the model given the start states.

    Args:
        model: The model to generate the trajectory from.
        model_args: The arguments used to generate the model.
        unnormalized_start_states: The initial states to start the trajectory from. These are un-normalized and will be normalized before passing to the model.
            batch_size x observation_dim
        only_execute_next_step: If True, only execute the next step of the trajectory. (like MPC)
        verbose: If True, print progress.
        batch_size: The batch size to use for generating the trajectories.
    """

    manifold=Product(sphere_dim = model_args.sphere_dim, torus_dim = model_args.torus_dim, euclidean_dim = model_args.euclidean_dim)

    # normalizer = LimitsNormalizer(params=model_args.normalization_params)

    normalizer = WrapManifold(manifold, params=model_args.normalization_params)

    start_states = normalizer.normalize(unnormalized_start_states)

    trajectories = []

    if verbose:
        import math
        total_num_batches = math.ceil(len(start_states) / batch_size)

    for idx in range(0, len(start_states), batch_size):
        if verbose:
            current_batch = math.ceil(idx / batch_size)

            print(f"[ scripts/visualize_trajectories ] Generating trajectories for batch {current_batch + 1}/{total_num_batches}" if total_num_batches > 1 else f"[ scripts/visualize_trajectories ] Generating trajectories")

        batch_start_states = start_states[idx: idx + batch_size]

        batch_trajectories = generate_trajectory_batch(
            batch_start_states, model, model_args, only_execute_next_step
        )

        trajectories.append(batch_trajectories)

    trajectories = np.concatenate(trajectories, axis=0)

    if return_normalize:
        return trajectories
    

    return unnormalize_trajectories(trajectories, model_args, verbose)


def unnormalize_trajectories(trajectories, model_args, verbose=False):
    """
    Process the trajectories to make them more interpretable.
    """

    manifold=Product(sphere_dim = model_args.sphere_dim, torus_dim = model_args.torus_dim, euclidean_dim = model_args.euclidean_dim)

    # normalizer = LimitsNormalizer(params=model_args.normalization_params)

    normalizer = WrapManifold(manifold, params=model_args.normalization_params)

    if verbose:
        print("[ utils/trajectory ] Processing trajectories")

    processed_trajectories = []

    for trajectory in trajectories:
        trajectory = normalizer.unnormalize(trajectory)
        # If value of trajectory[0] is greater than pi, then subtract 2pi from the trajectory
        # trajectory[trajectory[:, 0] > np.pi, 0] -= 2 * np.pi
        # trajectory[trajectory[:, 0] < -np.pi, 0] += 2 * np.pi

        # trajectory[trajectory[:, 0] < 0, 0] += 2 * np.pi

        processed_trajectories.append(trajectory)

    trajectories = np.array(processed_trajectories)

    return trajectories


def save_trajectories_image(trajectories, image_path, verbose=False, comparison_trajectories=None, show_traj_ends=False):
    """
    Visualize the trajectories generated by the model.
    """

    if verbose:
        print("[ utils/trajectory ] Visualizing trajectories")

    for idx in tqdm(range(trajectories.shape[0])):
        trajectory = trajectories[idx]

        plt.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            s=0.1,
            color="black",
            alpha=1,
            marker=".",
            label="Generated Trajectories",
        )

        if show_traj_ends:
            plt.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                s=10,
                color="black",
                alpha=1,
                marker="s",
            )
            plt.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                s=10,
                color="black",
                alpha=1,
                marker="x",
            )

    if comparison_trajectories is not None:
        if verbose:
            print("[ utils/trajectory ] Visualizing ground-truth trajectories")

        for idx in tqdm(range(len(comparison_trajectories))):
            trajectory = comparison_trajectories[idx]
            plt.scatter(
                trajectory[:, 0],
                trajectory[:, 1],
                s=0.1,
                color="red",
                alpha=1,
                marker="1",
                label="Ground Truth Trajectories",
            )

            if show_traj_ends:
                plt.scatter(
                    trajectory[0, 0],
                    trajectory[0, 1],
                    s=10,
                    color="red",
                    alpha=1,
                    marker="s",
                )
                plt.scatter(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    s=10,
                    color="red",
                    alpha=1,
                    marker="x",
                )

    plt.savefig(image_path)

    if verbose:
        print(f"[ utils/trajectory ] Trajectories saved at {image_path}")

def get_fnames_to_load(dataset_path, trajectories_path, num_trajs):
    indices_fpath = path.join(dataset_path, "shuffled_indices.txt")

    if path.exists(indices_fpath):
        with open(indices_fpath, "r") as f:
            fnames = f.readlines()
            fnames = [f.strip() for f in fnames]
            fnames = fnames[:num_trajs]

    else:
        print(f"[ utils/trajectory ] Could not find shuffled indices at {indices_fpath}. Generating new shuffled indices")
        all_fnames = listdir(trajectories_path)
        fnames = all_fnames.copy()
        shuffle(fnames)

        with open(indices_fpath, "w") as f:
            for fname in fnames:
                f.write(fname + "\n")

    fnames = fnames[:num_trajs]

    return fnames

def read_trajectory(sequence_path):
    with open(sequence_path, "r") as f:
        lines = f.readlines()

    trajectory = []

    for line in lines:
        state = line.split(",")
        state = [float(s) for s in state]

        trajectory.append(state)

    return trajectory

def load_trajectories(dataset, dataset_size=None, parallel=True, fnames=None):
    """
    load dataset from directory
    """
    dataset_path = path.join("data_trajectories", dataset)
    trajectories_path = path.join(dataset_path, "trajectories")

    if fnames is None:
        if dataset_size is None:
            try:
                fnames = listdir(trajectories_path)
            except FileNotFoundError:
                raise ValueError(f"Could not find dataset at {trajectories_path}")
        else:
            fnames = get_fnames_to_load(dataset_path, trajectories_path, dataset_size)

    trajectories = []

    print(f"[ datasets/sequence ] Loading trajectories from {trajectories_path}")

    fpaths = [path.join(trajectories_path, fname) for fname in fnames]

    if not parallel:
        for fpath in tqdm(fpaths):
            if not fpath.endswith(".txt"):
                continue
            trajectories.append(read_trajectory(fpath))
    else:
        import multiprocessing as mp

        with mp.Pool(cpu_count()) as pool:
            trajectories = list(
                tqdm(pool.imap(read_trajectory, fpaths), total=len(fpaths))
            )

    return np.array(trajectories, dtype=np.float32)


def get_trajectory_attractor_labels(final_states, attractors, attractor_threshold, invalid_label=-1):
    print("[ utils/trajectory ] Getting attractor labels for trajectories")

    attractor_states = attractors.keys()
    attractor_states = np.array(list(attractor_states))

    attractor_labels = attractors.values()
    attractor_labels = np.array(list(attractor_labels))
    attractor_labels = attractor_labels.reshape(-1, 1)

    # Compute the distance between the final states and each of the attractors
    distances = np.linalg.norm(final_states[:, None] - attractor_states, axis=2)

    min_distance = np.min(distances, axis=1)
    min_distance_idx = np.argmin(distances, axis=1)

    predicted_labels = np.zeros_like(min_distance)

    predicted_labels[min_distance <= attractor_threshold] = attractor_labels[min_distance_idx[min_distance < attractor_threshold]].flatten()
    predicted_labels[min_distance > attractor_threshold] = invalid_label

    return predicted_labels


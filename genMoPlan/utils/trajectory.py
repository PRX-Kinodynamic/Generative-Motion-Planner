from tqdm import tqdm
import matplotlib.pyplot as plt
from os import cpu_count, path, listdir
from random import shuffle

import torch
import numpy as np

from genMoPlan.datasets.normalization import *
from genMoPlan.models.generative.base import GenerativeModel

def _get_normalizer_params(model_args):
    normalizer_params = None

    if hasattr(model_args, "normalizer_params"):
        normalizer_params = model_args.normalizer_params["trajectory"]
    elif hasattr(model_args, "normalization_params"):
        normalizer_params = model_args.normalization_params
    else:
        raise ValueError("Normalizer params not found in model_args")

    return normalizer_params

def _generate_trajectory_batch(start_states: np.ndarray, model: GenerativeModel, model_args: dict, max_path_length: int, only_execute_next_step: bool = False, conditional_sample_kwargs: dict = {}, only_return_final_states: bool = False, verbose: bool = True):
    batch_size = len(start_states)

    current_states = torch.tensor(start_states, dtype=torch.float32).to(
        model_args.device
    )
    current_idx = model_args.history_length
    prediction_length = model_args.horizon_length if not only_execute_next_step else 1

    if not only_return_final_states:
        trajectories = np.zeros((batch_size, max_path_length, model_args.observation_dim))
        trajectories[:, 0] = np.array(start_states)
    else:
        trajectories = None
    
    with tqdm(total=max_path_length - model_args.history_length, disable=not verbose) as pbar:
        while current_idx < max_path_length:
            conditions = {0: current_states}

            next_trajs = model.forward(conditions, verbose=False, return_chain=False, **conditional_sample_kwargs).trajectories

            slice_path_length = min(prediction_length, max_path_length - current_idx)
            next_trajs = next_trajs[:, model_args.history_length: model_args.history_length + slice_path_length]

            # Adding the next states to the trajectory
            if not only_return_final_states:
                trajectories[:, current_idx: current_idx + slice_path_length] = next_trajs.cpu().numpy()

            current_states = next_trajs[:, -1]
            current_idx += slice_path_length

            # Free memory
            del next_trajs
            if next(model.parameters()).device.type == "cuda" and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.update(slice_path_length)

    if only_return_final_states:
        return current_states.cpu().detach().numpy()

    return trajectories


def generate_trajectories(
    model, model_args, unnormalized_start_states, max_path_length, only_execute_next_step: bool = False, verbose: bool = True, batch_size: int = 5000, conditional_sample_kwargs: dict = {}, only_return_final_states: bool = False
):
    """
    Generate a trajectory from the model given the start states.

    Args:
        model: The model to generate the trajectory from.
        model_args: The arguments used to generate the model.
        unnormalized_start_states: The initial states to start the trajectory from. These are un-normalized and will be normalized before passing to the model.
            batch_size x observation_dim
        max_path_length: The maximum length of the trajectory to generate.
        only_execute_next_step: If True, only execute the next step of the trajectory. (like MPC)
        verbose: If True, print progress.
        batch_size: The batch size to use for generating the trajectories.
        only_return_final_states: If True, only return the final states of the trajectories.
    """
    normalizer_class = eval(model_args.trajectory_normalizer)

    normalizer: Normalizer = normalizer_class(params=_get_normalizer_params(model_args))

    start_states = normalizer.normalize(unnormalized_start_states)

    if only_return_final_states:
        final_states = np.zeros_like(start_states)
    else:
        trajectories = []

    if verbose:
        import math
        total_num_batches = math.ceil(len(start_states) / batch_size)

    for idx in range(0, len(start_states), batch_size):
        if verbose:
            current_batch = math.ceil(idx / batch_size)

            print(f"[ utils/trajectory ] Generating trajectories for batch {current_batch + 1}/{total_num_batches}" if total_num_batches > 1 else f"[ utils/trajectory ] Generating trajectories")

        batch_start_states = start_states[idx: idx + batch_size]

        results = _generate_trajectory_batch(
            batch_start_states, model, model_args, max_path_length, only_execute_next_step, conditional_sample_kwargs, only_return_final_states=only_return_final_states, verbose=verbose
        )

        if only_return_final_states:
            final_states[idx:idx+batch_size] = results
        else:
            trajectories.append(results)
            
        # Free memory
        if next(model.parameters()).device.type == "cuda" and hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not only_return_final_states:
        trajectories = np.concatenate(trajectories, axis=0)
    
    if only_return_final_states:
        return process_states(final_states, normalizer, verbose)

    return process_trajectories(trajectories, normalizer, verbose)

def process_states(states, normalizer, verbose=False):
    """
    Process the states
    - Un-normalize the states
    - Move the states from [-2pi, 2pi] to [-pi, pi]
    """
    states = normalizer.unnormalize(states)

    states[states[:, 0] > np.pi, 0] -= 2 * np.pi
    states[states[:, 0] < -np.pi, 0] += 2 * np.pi

    return states


def process_trajectories(trajectories, normalizer, verbose=False):
    """
    Process the trajectories
    - Un-normalize the trajectories
    - Move the trajectories from [-2pi, 2pi] to [-pi, pi]
    """
    processed_trajectories = []

    itr = tqdm(trajectories, desc="[ utils/trajectory ] Processing trajectories") if verbose else trajectories

    for trajectory in itr:
        trajectory = normalizer.unnormalize(trajectory)
        # If value of trajectory[0] is greater than pi, then subtract 2pi from the trajectory
        trajectory[trajectory[:, 0] > np.pi, 0] -= 2 * np.pi
        trajectory[trajectory[:, 0] < -np.pi, 0] += 2 * np.pi

        processed_trajectories.append(trajectory)

    trajectories = np.array(processed_trajectories)

    return trajectories


def plot_trajectories(trajectories, image_path=None, verbose=False, comparison_trajectories=None, show_traj_ends=False, return_plot=False):
    """
    Visualize the trajectories generated by the model.
    """

    # Reshape trajectories from n x length x dim to (n x length) x dim
    n, length, dim = trajectories.shape
    trajectories_reshaped = trajectories.reshape(-1, dim)

    plt.scatter(
        trajectories_reshaped[:, 0],
        trajectories_reshaped[:, 1],
        s=0.1,
        color="black",
        alpha=1,
        marker=".",
        label="Generated Trajectories",
    )

    if show_traj_ends:
        plt.scatter(
            trajectories[:, 0, 0],
            trajectories[:, 0, 1],
            s=10,
            color="black",
            alpha=1,
            marker="s",
        )

        plt.scatter(
            trajectories[:, -1, 0],
            trajectories[:, -1, 1],
            s=10,
            color="black",
            alpha=1,
            marker="x",
        )
    if comparison_trajectories is not None:
        comparison_trajectories_reshaped = comparison_trajectories.reshape(-1, dim)

        plt.scatter(
            comparison_trajectories_reshaped[:, 0],
            comparison_trajectories_reshaped[:, 1],
            s=0.1,
            color="red",
            alpha=1,
            marker="1",
            label="Ground Truth Trajectories",
        )

        if show_traj_ends:
            plt.scatter(
                comparison_trajectories[:, 0, 0],
                comparison_trajectories[:, 0, 1],
                s=10,
                color="red",
                alpha=1,
                marker="s",
            )

            plt.scatter(
                comparison_trajectories[:, -1, 0],
                comparison_trajectories[:, -1, 1],
                s=10,
                color="red",
                alpha=1,
                marker="x",
            )

    if image_path is not None:
        plt.savefig(image_path)
        plt.close()

        if verbose:
            print(f"[ utils/trajectory ] Trajectories saved at {image_path}")

    if return_plot:
        return plt



def get_fnames_to_load(dataset_path, trajectories_path, num_trajs=None, load_reverse=False):
    indices_fpath = path.join(dataset_path, "shuffled_indices.txt")

    if path.exists(indices_fpath):
        with open(indices_fpath, "r") as f:
            fnames = f.readlines()
            fnames = [f.strip() for f in fnames]

    else:
        print(f"[ utils/trajectory ] Could not find shuffled indices at {indices_fpath}. Generating new shuffled indices")
        all_fnames = listdir(trajectories_path)
        fnames = all_fnames.copy()
        shuffle(fnames)

        with open(indices_fpath, "w") as f:
            for fname in fnames:
                f.write(fname + "\n")

    if num_trajs is not None:
        if not load_reverse:
            fnames = fnames[:num_trajs]
        else:
            fnames = fnames[-num_trajs:]

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


def load_trajectories(dataset, dataset_size=None, parallel=True, fnames=None, load_reverse=False):
    """
    load dataset from directory
    """
    dataset_path = path.join("data_trajectories", dataset)
    trajectories_path = path.join(dataset_path, "trajectories")

    if fnames is None:
        fnames = get_fnames_to_load(dataset_path, trajectories_path, dataset_size, load_reverse)

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


def get_trajectory_attractor_labels(final_states: np.ndarray, attractors: dict, attractor_dist_threshold: float, invalid_label: int = -1, verbose: bool = True):
    if verbose:
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

    predicted_labels[min_distance <= attractor_dist_threshold] = attractor_labels[min_distance_idx[min_distance < attractor_dist_threshold]].flatten()
    predicted_labels[min_distance > attractor_dist_threshold] = invalid_label

    return predicted_labels


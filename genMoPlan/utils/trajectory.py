from os import cpu_count, path
from typing import List, Union, Sequence, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from genMoPlan.utils import get_data_trajectories_path


def process_angles(data, angle_indices=None):
    """
    Process the angles of the data to normalize them to [-pi, pi] range
    """
    if angle_indices is None:
        raise ValueError("angle_indices must be provided")

    data = data.copy()

    for idx in angle_indices:
        # First ensure angles are in [0, 2pi] range
        data[..., idx] = np.mod(data[..., idx], 2 * np.pi)
        # Then convert from [0, 2pi] to [-pi, pi]
        data[..., idx][data[..., idx] > np.pi] -= 2 * np.pi

    return data


def plot_trajectories(
    trajectories: Union[np.ndarray, Sequence[np.ndarray]],
    image_path: Optional[str] = None,
    verbose: bool = False,
    comparison_trajectories: Union[np.ndarray, Sequence[np.ndarray], None] = None,
    show_traj_ends: bool = False,
    return_plot: bool = False,
):
    """
    Visualize 2-D trajectories.

    The function now supports both a single ``numpy.ndarray`` of shape
    (n_trajectories, length, dim) as well as a ``List`` of numpy arrays of
    shape (length_i, dim). Handling the list directly avoids an unnecessary
    concatenation when the caller already maintains the trajectories in that
    format.
    """

    # Helper that flattens trajectories and extracts start/end points while
    # supporting both list- and ndarray-based inputs.
    def _flatten(trajs):
        if trajs is None:
            return None, None, None

        # If we already have a single ndarray, life is easy.
        if isinstance(trajs, np.ndarray):
            n, length, dim = trajs.shape
            flat = trajs.reshape(-1, dim)
            starts = trajs[:, 0]
            ends = trajs[:, -1]
            return flat, starts, ends

        # Otherwise assume a sequence (e.g. list) of ndarrays with matching dim.
        dim = trajs[0].shape[-1]
        flat = np.concatenate(trajs, axis=0)
        starts = np.array([t[0] for t in trajs])
        ends = np.array([t[-1] for t in trajs])
        return flat, starts, ends

    traj_flat, traj_starts, traj_ends = _flatten(trajectories)
    if traj_flat is None:
        raise ValueError("`trajectories` cannot be None")

    # Plot generated trajectories
    plt.scatter(
        traj_flat[:, 0],
        traj_flat[:, 1],
        s=0.1,
        color="black",
        alpha=1,
        marker=".",
        label="Generated Trajectories",
    )

    if show_traj_ends:
        plt.scatter(
            traj_starts[:, 0],
            traj_starts[:, 1],
            s=10,
            color="black",
            alpha=1,
            marker="s",
        )
        plt.scatter(
            traj_ends[:, 0],
            traj_ends[:, 1],
            s=10,
            color="black",
            alpha=1,
            marker="x",
        )

    if comparison_trajectories is not None:
        comp_flat, comp_starts, comp_ends = _flatten(comparison_trajectories)
        plt.scatter(
            comp_flat[:, 0],
            comp_flat[:, 1],
            s=0.1,
            color="red",
            alpha=1,
            marker="1",
            label="Ground Truth Trajectories",
        )
        if show_traj_ends:
            plt.scatter(
                comp_starts[:, 0],
                comp_starts[:, 1],
                s=10,
                color="red",
                alpha=1,
                marker="s",
            )
            plt.scatter(
                comp_ends[:, 0],
                comp_ends[:, 1],
                s=10,
                color="red",
                alpha=1,
                marker="x",
            )

    if image_path is not None:
        plt.savefig(image_path)
        if verbose:
            print(f"[ utils/trajectory ] Trajectories saved at {image_path}")

    if return_plot:
        return plt

    plt.close()


def get_fnames_to_load(
    dataset_path,
    trajectories_path=None,
    num_trajs=None,
    load_reverse=False,
    shuffled_indices_fname="shuffled_indices.txt",
):
    if trajectories_path is None:
        trajectories_path = path.join(dataset_path, "trajectories")

    indices_fpath = path.join(dataset_path, "train_test_splits", shuffled_indices_fname)

    if not path.exists(indices_fpath):
        raise FileNotFoundError(
            f"[ utils/trajectory ] Could not find shuffled indices at {indices_fpath}"
        )

    with open(indices_fpath, "r") as f:
        fnames = f.readlines()
        fnames = [f.strip() for f in fnames]

    if num_trajs is not None:
        if not load_reverse:
            fnames = fnames[:num_trajs]
        else:
            fnames = fnames[-num_trajs:]

    return fnames


def _read_trajectories_from_fpaths(
    read_trajectory_fn, trajectories_path, fnames, parallel=True
):
    fpaths = [path.join(trajectories_path, fname) for fname in fnames]
    if not parallel:
        trajectories = []
        for fpath in tqdm(fpaths):
            if not fpath.endswith(".txt"):
                continue
            trajectories.append(read_trajectory_fn(fpath))
    else:
        import multiprocessing as mp

        with mp.Pool(cpu_count()) as pool:
            trajectories = list(
                tqdm(pool.imap(read_trajectory_fn, fpaths), total=len(fpaths))
            )

    trajectories = [trajectory for trajectory in trajectories if trajectory is not None]

    return trajectories


def load_trajectories(
    dataset,
    read_trajectory_fn,
    dataset_size=None,
    parallel=True,
    fnames=None,
    load_reverse=False,
    shuffled_indices_fname="shuffled_indices.txt",
) -> List[np.ndarray]:
    """
    load dataset from directory
    """
    dataset_path = path.join(get_data_trajectories_path(), dataset)
    trajectories_path = path.join(dataset_path, "trajectories")

    if fnames is None:
        fnames = get_fnames_to_load(
            dataset_path, trajectories_path, dataset_size, load_reverse,
            shuffled_indices_fname=shuffled_indices_fname,
        )

    trajectories = []

    print(f"[ datasets/sequence ] Loading trajectories from {trajectories_path}")

    trajectories = _read_trajectories_from_fpaths(
        read_trajectory_fn, trajectories_path, fnames, parallel=parallel
    )
    trajectories = [
        np.array(trajectory, dtype=np.float32)
        for trajectory in trajectories
        if trajectory is not None
    ]

    return trajectories

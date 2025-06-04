from collections import namedtuple
from enum import Enum
from functools import partial
from os import path, listdir
import multiprocessing as mp
from random import shuffle
from typing import List, Tuple
from tqdm import tqdm
import torch
import numpy as np

try:
    import numba
    has_numba = True
except ImportError:
    has_numba = False

from genMoPlan.datasets.normalization import *
from genMoPlan.datasets.utils import apply_padding, compute_actual_length
from genMoPlan.utils import read_trajectories_from_fpaths


Batch = namedtuple("Batch", "trajectory conditions query")

class TrajectoryBoundsType(Enum):
    LT_2PI = 0
    GT_N2PI = 1
    GT_0 = 2
    LT_0 = 3

class HorizonVariationType(Enum):
    BASE = 0
    INVERTED = 1
    PLUS_2PI = 2
    MINUS_2PI = 3
    INV_PLUS_2PI = 4
    INV_MINUS_2PI = 5
    PLUS_4PI = 6
    MINUS_4PI = 7
    INV_PLUS_4PI = 8
    INV_MINUS_4PI = 9

TWO_PI = 2 * np.pi
FOUR_PI = 4 * np.pi

def _get_query_fnames(trajectories_path):
    fnames = listdir(trajectories_path)

    unique_fnames = set()

    for fname in fnames:
        fsplit = fname.split("_")
        query_fname = '_'.join(fsplit[:-1])

        if path.exists(path.join(trajectories_path, f"{query_fname}_0.txt")):
            unique_fnames.add(query_fname)

    return list(unique_fnames)


def _load_trajectories(dataset, observation_dim, dataset_size, only_optimal, velocity_limit, load_reverse=False, parallel=True):
    dataset_path = path.join("data_trajectories", dataset)
    trajectories_path = path.join(dataset_path, "trajectories")
    fnames_fpath = path.join(dataset_path, f"shuffled_indices_{'optimal' if only_optimal else 'all'}.txt")

    if not path.exists(fnames_fpath):
        if only_optimal:
            query_fnames = _get_query_fnames(trajectories_path)
            fnames = [f'{fname}_0.txt' for fname in query_fnames]
        else:
            fnames = listdir(trajectories_path)

        shuffle(fnames)

        with open(fnames_fpath, "w") as f:
            for fname in fnames:
                f.write(fname + "\n")
    else:
        with open(fnames_fpath, "r") as f:
            fnames = f.readlines()
            fnames = [f.strip() for f in fnames]

        if load_reverse:
            fnames = fnames[::-1]

    if dataset_size is not None:
        fnames = fnames[:dataset_size]

    loaded_trajectories = read_trajectories_from_fpaths(trajectories_path, fnames, observation_dim, delimiter=" ", ignore_empty_lines=True, parallel=parallel)

    original_traj_count = len(loaded_trajectories)

    filtered_trajectories = []

    for trajectory in loaded_trajectories:
        if np.all(np.abs(trajectory[:, 2]) <= velocity_limit) and np.all(np.abs(trajectory[:, 3]) <= velocity_limit):
            filtered_trajectories.append(trajectory)

    trajectories = filtered_trajectories

    filtered_traj_count = len(trajectories)

    print(f"[ datasets/acrobot ] Filtered {original_traj_count - filtered_traj_count} trajectories out of {original_traj_count}")

    return trajectories


def _get_trajectory_bound_flags(trajectory, use_original_data):
    bound_flags = np.zeros((len(trajectory), len(TrajectoryBoundsType)))

    if use_original_data:
        return bound_flags

    # remaining_bound_types = set(TrajectoryBoundsType)

    # for i in range(len(trajectory)):
    #     if TrajectoryBoundsType.LT_2PI in remaining_bound_types and trajectory[i, 0] < TWO_PI and trajectory[i, 1] < TWO_PI:
    #         bound_flags[i:, TrajectoryBoundsType.LT_2PI.value] = 1
    #         remaining_bound_types.remove(TrajectoryBoundsType.LT_2PI)

    #     if TrajectoryBoundsType.GT_N2PI in remaining_bound_types and trajectory[i, 0] > -TWO_PI and trajectory[i, 1] > -TWO_PI:
    #         bound_flags[i:, TrajectoryBoundsType.GT_N2PI.value] = 1
    #         remaining_bound_types.remove(TrajectoryBoundsType.GT_N2PI)

    #     if TrajectoryBoundsType.GT_0 in remaining_bound_types and trajectory[i, 0] > 0 and trajectory[i, 1] > 0:
    #         bound_flags[i:, TrajectoryBoundsType.GT_0.value] = 1
    #         remaining_bound_types.remove(TrajectoryBoundsType.GT_0)

    #     if TrajectoryBoundsType.LT_0 in remaining_bound_types and trajectory[i, 0] < 0 and trajectory[i, 1] < 0:
    #         bound_flags[i:, TrajectoryBoundsType.LT_0.value] = 1
    #         remaining_bound_types.remove(TrajectoryBoundsType.LT_0)

    #     if len(remaining_bound_types) == 0:
    #         break

    return bound_flags

def _get_possible_variation_types(bound_flags):
    """Get possible variation types based on bound flags."""
    possible_variation_types = [
        HorizonVariationType.BASE.value,
        HorizonVariationType.INVERTED.value
    ]

    has_lt_2pi = False
    has_gt_n2pi = False
    has_gt_0 = False
    has_lt_0 = False

    for flag_val in bound_flags:
        if flag_val == TrajectoryBoundsType.LT_2PI.value:
            has_lt_2pi = True
        elif flag_val == TrajectoryBoundsType.GT_N2PI.value:
            has_gt_n2pi = True
        elif flag_val == TrajectoryBoundsType.GT_0.value:
            has_gt_0 = True
        elif flag_val == TrajectoryBoundsType.LT_0.value:
            has_lt_0 = True

    if has_lt_2pi:
        possible_variation_types.append(HorizonVariationType.PLUS_2PI.value)
        possible_variation_types.append(HorizonVariationType.INV_PLUS_2PI.value)
    
    if has_gt_n2pi:
        possible_variation_types.append(HorizonVariationType.INV_MINUS_2PI.value)
        possible_variation_types.append(HorizonVariationType.MINUS_2PI.value)
    
    if has_gt_0:
        possible_variation_types.append(HorizonVariationType.MINUS_4PI.value)
        possible_variation_types.append(HorizonVariationType.INV_MINUS_4PI.value)

    if has_lt_0:
        possible_variation_types.append(HorizonVariationType.PLUS_4PI.value)
        possible_variation_types.append(HorizonVariationType.INV_PLUS_4PI.value)

    return possible_variation_types

def _make_trajectory_indices(actual_history_length, actual_horizon_length, stride, use_history_padding, use_horizon_padding, use_original_data, bound_flags_for_path, traj_length, path_ind) -> List[Tuple]:
    indices = []

    min_history_elements = 1 if use_history_padding else actual_history_length
    min_horizon_elements = 0 if use_horizon_padding else actual_horizon_length

    if min_history_elements + min_horizon_elements > traj_length:
        return indices

    max_history_end = traj_length - min_horizon_elements - stride + 1
    min_history_end = min_history_elements
    
    for history_end in range(min_history_end, max_history_end + 1):
        history_start = max(0, history_end - actual_history_length)
        horizon_start = history_end + stride - 1
        horizon_end = min(horizon_start + actual_horizon_length, traj_length)

        current_bound_flags_indices = np.where(bound_flags_for_path[history_start] == 1)[0]

        if use_original_data:
            possible_variation_types = [
                HorizonVariationType.BASE.value,
            ]
        else:
            possible_variation_types = [
                HorizonVariationType.BASE.value,
                HorizonVariationType.INVERTED.value,
            ]
            # possible_variation_types = _get_possible_variation_types(current_bound_flags_indices)

        for variation_type_val in possible_variation_types:
            indices.append((path_ind, history_start, history_end, horizon_start, horizon_end, variation_type_val))

    return indices

# Apply numba JIT if available
if has_numba:
    _get_possible_variation_types = numba.njit(_get_possible_variation_types)
    _make_trajectory_indices = numba.njit(_make_trajectory_indices)

def _make_indices(path_lengths, bound_flags, history_length, use_history_padding, horizon_length, use_horizon_padding, stride, use_original_data, parallel=True):
    if use_history_padding and use_horizon_padding:
        raise ValueError("Cannot use both history and horizon padding")

    indices = []
    
    actual_horizon_length = compute_actual_length(horizon_length, stride)
    actual_history_length = compute_actual_length(history_length, stride)

    print(f"[ datasets/acrobot ] Actual history length: {actual_history_length}, Actual horizon length: {actual_horizon_length}")
    print(f"[ datasets/acrobot ] Preparing indices for dataset")

    args_list = []
    total_orig_indices_count = 0

    for i in range(len(path_lengths)):
        args_list.append(
            (actual_history_length, actual_horizon_length, stride,use_history_padding, use_horizon_padding, use_original_data,bound_flags[i], path_lengths[i], i)
        )

        min_history_elements = 1 if use_history_padding else actual_history_length
        min_horizon_elements = 0 if use_horizon_padding else actual_horizon_length

        if min_history_elements + min_horizon_elements <= path_lengths[i]:
            total_orig_indices_count += path_lengths[i] - min_horizon_elements - min_history_elements + 1

    if not parallel:
        for args in tqdm(args_list, total=len(args_list)):
            indices.extend(_make_trajectory_indices(*args))
    else:
        with mp.get_context("fork").Pool(mp.cpu_count()) as pool:
            results = list(pool.starmap(_make_trajectory_indices, args_list))
            for result_list in results:
                indices.extend(result_list)


    if len(indices) == 0:
        raise ValueError("No valid trajectories found for the dataset")
    
    print(f'[ datasets/acrobot ] Data Augmented by {len(indices) / total_orig_indices_count:.2f}x')

    return indices


def _apply_variation_type(history, horizon, variation_type):
    if isinstance(variation_type, int):
        variation_type = HorizonVariationType(variation_type)

    if variation_type == HorizonVariationType.BASE:
        pass
    elif variation_type == HorizonVariationType.INVERTED:
        horizon = -horizon
        history = -history
    elif variation_type == HorizonVariationType.PLUS_2PI:
        horizon[:, :2] += TWO_PI
        history[:, :2] += TWO_PI
    elif variation_type == HorizonVariationType.PLUS_4PI:
        horizon[:, :2] += FOUR_PI
        history[:, :2] += FOUR_PI
    elif variation_type == HorizonVariationType.MINUS_2PI:
        horizon[:, :2] -= TWO_PI
        history[:, :2] -= TWO_PI
    elif variation_type == HorizonVariationType.MINUS_4PI:
        horizon[:, :2] -= FOUR_PI
        history[:, :2] -= FOUR_PI
    elif variation_type == HorizonVariationType.INV_PLUS_2PI:
        horizon[:, :2] += TWO_PI
        history[:, :2] += TWO_PI
        horizon = -horizon
        history = -history
    elif variation_type == HorizonVariationType.INV_MINUS_2PI:
        horizon[:, :2] -= TWO_PI
        history[:, :2] -= TWO_PI
        horizon = -horizon
        history = -history
    elif variation_type == HorizonVariationType.INV_PLUS_4PI:
        horizon[:, :2] += FOUR_PI
        history[:, :2] += FOUR_PI
        horizon = -horizon
        history = -history
    elif variation_type == HorizonVariationType.INV_MINUS_4PI:
        horizon[:, :2] -= FOUR_PI
        history[:, :2] -= FOUR_PI
        horizon = -horizon
        history = -history
    else:
        raise ValueError(f"Invalid variation type: {variation_type}")

    return history, horizon


class AcrobotDataset(torch.utils.data.Dataset):
    normed_trajectories = None

    def __init__(
        self,
        dataset: str = None,
        horizon_length: int = 13,
        history_length: int = 3,
        stride: int = 10,
        observation_dim: int = None,
        trajectory_normalizer: str = "LimitsNormalizer",
        normalizer_params: dict = {},
        dataset_size: int = None,
        use_horizon_padding: bool = False,
        use_history_padding: bool = False,
        is_history_conditioned: bool = True, # Otherwise it is provided as a query that is not predicted by the model
        is_validation: bool = False,
        only_optimal: bool = True,
        use_original_data: bool = False,
        velocity_limit: float = float("inf"),
        **kwargs,
    ):
        self.horizon_length = horizon_length
        self.history_length = history_length
        self.stride = stride
        self.use_horizon_padding = use_horizon_padding
        self.use_history_padding = use_history_padding
        self.is_history_conditioned = is_history_conditioned
        self.observation_dim = observation_dim
        
        trajectories, bound_flags = self._load_data(
            dataset,
            dataset_size, 
            is_validation,
            only_optimal,
            velocity_limit,
            use_original_data,
        )

        traj_lengths = [len(trajectory) for trajectory in trajectories]
        
        self.indices = _make_indices(
            traj_lengths, bound_flags, self.history_length, self.use_history_padding, self.horizon_length, self.use_horizon_padding, self.stride, use_original_data, parallel=True
        )

        self.bound_flags = bound_flags
        self.n_episodes = len(trajectories)
        self.traj_lengths = traj_lengths

        self.normed_trajectories = self._normalize(
            trajectories, 
            trajectory_normalizer, 
            normalizer_params
        )

    def _load_data(self, dataset, dataset_size, is_validation, only_optimal, velocity_limit, use_original_data):
        trajectories = _load_trajectories(dataset, self.observation_dim, dataset_size, only_optimal, velocity_limit, load_reverse=is_validation, parallel=True)

        bound_flags = self._get_bound_flags(trajectories, use_original_data)
        
        return trajectories, bound_flags
    
    def _get_bound_flags(self, trajectories, use_original_data):
        print("[ data_trajectories/acrobot/data_loader ] Getting bound flags")

        args_list = [
            (trajectory, use_original_data) for trajectory in trajectories
        ]
        with mp.Pool(mp.cpu_count()) as pool:
            # Ensure _get_trajectory_bound_flags returns NumPy arrays
            bound_flags_list = list(tqdm(pool.starmap(_get_trajectory_bound_flags, args_list), total=len(args_list)))

        # Ensure each element is a NumPy array for Numba compatibility later
        return [np.array(bf, dtype=np.int8) for bf in bound_flags_list] # Use int8 for flags to save memory

    def _normalize(self, trajectories, trajectory_normalizer=None, normalizer_params=None):
        """
        normalize fields that will be predicted

        First, aggregate all trajectories into a single array
        Then, normalize the aggregated array
        Then, split the normalized array back into individual trajectories. 
        """
        if trajectory_normalizer is None:
            return trajectories
        
        print(f"[ datasets/acrobot ] Normalizing trajectories")

        all_trajectories = np.concatenate(trajectories, axis=0)
        
        if type(trajectory_normalizer) == str:
            trajectory_normalizer = eval(trajectory_normalizer)
        trajectory_normalizer = trajectory_normalizer(X=trajectories, params=normalizer_params["trajectory"])
        normed_all_trajectories = trajectory_normalizer(all_trajectories)

        # Split all trajectories into individual trajectories
        traj_lengths = [len(traj) for traj in trajectories]
        trajectories = []
        traj_start_idx = 0
        
        for traj_length in traj_lengths:
            traj_end_idx = traj_start_idx + traj_length
            normed_traj = normed_all_trajectories[traj_start_idx:traj_end_idx]
    
            trajectories.append(torch.FloatTensor(normed_traj))
            traj_start_idx = traj_end_idx
        
        return trajectories

    def get_conditions(self, history):
        """
        conditions on current observation for planning
        """
        return dict(enumerate(history))

    def get_query(self, history):
        """
        query on current observation for planning
        """
        if self.is_history_conditioned:
            return torch.zeros(0)
        else:
            return history
        
    def get_trajectory(self, history, horizon):
        """
        trajectory on current observation for planning
        """
        if self.is_history_conditioned:
            return torch.cat([history, horizon], dim=0)
        else:
            return horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, history_start, history_end, horizon_start, horizon_end, variation_type = self.indices[idx]

        # Extract the actual available data
        trajectory_data = self.normed_trajectories[path_ind]
        
        # Get actual data for history and horizon
        history = trajectory_data[history_start:history_end:self.stride]
        horizon = trajectory_data[horizon_start:horizon_end:self.stride]

        history, horizon = _apply_variation_type(history, horizon, variation_type)

        assert len(history) > 0, "History is empty"

        if self.use_history_padding:
            history = apply_padding(history, self.history_length, pad_left=True)

        if self.use_horizon_padding:
            if len(horizon) > 0:
                pad_value = horizon[-1]
            else:
                pad_value = history[-1]

            horizon = apply_padding(horizon, self.horizon_length, pad_left=False, pad_value=pad_value)

        assert len(history) == self.history_length, f"History length is {len(history)}, expected {self.history_length}"
        assert len(horizon) == self.horizon_length, f"Horizon length is {len(horizon)}, expected {self.horizon_length}"
        

        trajectory = self.get_trajectory(history, horizon)
        conditions = self.get_conditions(history)
        query = self.get_query(history)

        batch = Batch(trajectory=trajectory, conditions=conditions, query=query)

        return batch

from collections import defaultdict, namedtuple
import numpy as np
import torch
from tqdm import tqdm

from genMoPlan.utils.parallel import parallelize_toggle


NONE_TENSOR = torch.zeros(0, dtype=torch.float32, device='cpu')
EMPTY_DICT = {}

Index = namedtuple("Index", "path_ind history_start history_end horizon_start horizon_end")
DataSample = namedtuple("DataSample", "trajectory conditions global_query local_query", defaults=(EMPTY_DICT, NONE_TENSOR, NONE_TENSOR))

def compute_actual_length(length, stride):
    return 1 + (length - 1) * stride

def _make_indices_for_single_trajectory(i, traj_length, actual_history_length, actual_horizon_length, stride, use_history_padding, use_horizon_padding):
    min_history_elements = 1 if use_history_padding else actual_history_length
    min_horizon_elements = 0 if use_horizon_padding else actual_horizon_length

    if min_history_elements + min_horizon_elements > traj_length:
        return None

    max_history_end = traj_length - min_horizon_elements
    min_history_end = min_history_elements

    unique_indices = set()

    traj_indices = []
    
    for history_end in range(min_history_end, max_history_end + 1):
        history_start = max(0, history_end - actual_history_length)
        horizon_start = history_end + stride - 1
        horizon_end = min(horizon_start + actual_horizon_length, traj_length)

        history_indices = tuple(range(history_start, history_end, stride))
        horizon_indices = tuple(range(horizon_start, horizon_end, stride))

        index_key = (history_indices, horizon_indices)

        if index_key in unique_indices:
            continue

        unique_indices.add(index_key)
        
        traj_indices.append(Index(i, history_start, history_end, horizon_start, horizon_end))

    return traj_indices
    

def make_indices(path_lengths, history_length, use_history_padding, horizon_length, use_horizon_padding, stride, parallel=True):
    if use_history_padding and use_horizon_padding:
        raise ValueError("Cannot use both history and horizon padding")

    indices = []
    skip_count = 0
    
    actual_horizon_length = compute_actual_length(horizon_length, stride)
    actual_history_length = compute_actual_length(history_length, stride)

    print(f"[ datasets/utils ] Actual history length: {actual_history_length}, Actual horizon length: {actual_horizon_length}")

    args_list = [
        (i, traj_length, actual_history_length, actual_horizon_length, stride, use_history_padding, use_horizon_padding)
        for i, traj_length in enumerate(path_lengths)
    ]

    results = parallelize_toggle(
        _make_indices_for_single_trajectory,
        args_list,
        parallel=parallel,
        show_progress=True,
        desc="[ datasets/utils ] Preparing indices for dataset",
    )

    for result in results:
        if result is None:
            skip_count += 1
        else:
            indices.extend(result)

    if len(indices) == 0:
            raise ValueError("No valid trajectories found for the dataset")

    if skip_count > 0:
        print(f"[ datasets/trajectory ] Skipped {skip_count} trajectories are shorter than (history length + horizon length)")

    return indices

def apply_padding(trajectory, length, pad_left=True, pad_value=None):
    if len(trajectory) == length:
        return trajectory
    
    if pad_value is None and len(trajectory) == 0:
        raise ValueError("Cannot pad empty trajectory with no pad value")
    
    if pad_value is None:
        if pad_left:
            pad_value = trajectory[0]
        else:
            pad_value = trajectory[-1]

    padding = pad_value.repeat(length - len(trajectory), 1)

    if pad_left:
        return torch.cat([padding, trajectory], dim=0)
    else:
        return torch.cat([trajectory, padding], dim=0)
    
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from typing import List, Dict, Union, Tuple, Any, Sequence

import numpy as np
import torch
import math

def _add_new_unwrapped_state_trajectories(angle_indices, original_trajectory: np.ndarray) -> List[np.ndarray]:
    all_trajectories = []
    
    # Check if the entire trajectory remains within [0, π] for all specified angle indices
    if np.all(np.all((original_trajectory[:, angle_indices] >= 0) & (original_trajectory[:, angle_indices] <= np.pi), axis=0)):
        new_trajectory = original_trajectory.copy()
        new_trajectory[:, angle_indices] -= 2 * np.pi
        all_trajectories.append(new_trajectory)

    # Check if the entire trajectory remains within [-π, 0] for all specified angle indices
    if np.all(np.all((original_trajectory[:, angle_indices] >= -np.pi) & (original_trajectory[:, angle_indices] <= 0), axis=0)):
        new_trajectory = original_trajectory.copy()
        new_trajectory[:, angle_indices] += 2 * np.pi
        all_trajectories.append(new_trajectory)

    return all_trajectories

def _process_trajectory_and_plan(data, index, angle_indices, process_fn):
    """
    Helper function to process a trajectory and its corresponding plan.
    
    Args:
        data: A tuple of (trajectory, plan) or just trajectory
        index: Index of the trajectory
        angle_indices: Indices of angles to process
        process_fn: Function to process the trajectory
        
    Returns:
        A list of tuples (new_trajectory, new_plan) or list of new_trajectories
    """
    if isinstance(data, tuple):
        trajectory, plan = data
        new_trajectories = process_fn(angle_indices, trajectory)
        if new_trajectories:
            return [(new_traj, plan.copy()) for new_traj in new_trajectories]
        else:
            return []
    else:
        trajectory = data
        return process_fn(angle_indices, trajectory)

def augment_unwrapped_state_data(data_or_trajectories, parallel=True, angle_indices: List[int] = [0], **kwargs):
    """
    Augment unwrapped state data by creating new copies of trajectories with angles modified.
    
    Args:
        data_or_trajectories: Either a dictionary with 'trajectories' and 'plans' keys, 
                             or a list of trajectory arrays
        parallel: Boolean flag to use parallel processing
        angle_indices: List of indices corresponding to angles
        
    Returns:
        Updated data dictionary or list of trajectories
    """
    print(f"[ utils/data_processing ] Augmenting unwrapped state data")
    
    is_dict_input = isinstance(data_or_trajectories, dict)
    
    if is_dict_input:
        trajectories = data_or_trajectories["trajectories"]
        plans = data_or_trajectories["plans"]
        trajectory_plan_pairs = list(zip(trajectories, plans))
        original_pairs = list(trajectory_plan_pairs)
    else:
        trajectories = data_or_trajectories
        original_trajectories = list(trajectories)
    
    # Get all combinations of angle indices
    from itertools import combinations
    angle_indices_combinations = []
    for r in range(1, len(angle_indices) + 1):
        angle_indices_combinations.extend(list(combinations(angle_indices, r)))
    
    if is_dict_input:
        updated_pairs = list(original_pairs)
    else:
        updated_trajectories = list(original_trajectories)
    
    for i, angle_index_combination in enumerate(angle_indices_combinations):
        angle_index_combination = list(angle_index_combination)
        print(f"[ utils/data_processing ] Processing angle combination {angle_index_combination} ({i+1}/{len(angle_indices_combinations)})")
        
        if is_dict_input:
            if not parallel:
                for idx, pair in enumerate(tqdm(original_pairs)):
                    new_pairs = _process_trajectory_and_plan(pair, idx, angle_index_combination, _add_new_unwrapped_state_trajectories)
                    updated_pairs.extend(new_pairs)
            else:
                with mp.Pool(mp.cpu_count()) as pool:
                    process_func = partial(_process_trajectory_and_plan, 
                                          angle_indices=angle_index_combination, 
                                          process_fn=_add_new_unwrapped_state_trajectories)
                    new_pairs_lists = list(
                        tqdm(
                            pool.starmap(
                                process_func, 
                                [(pair, idx) for idx, pair in enumerate(original_pairs)]
                            ),
                            total=len(original_pairs),
                        )
                    )
                    for new_pairs in new_pairs_lists:
                        updated_pairs.extend(new_pairs)
        else:
            if not parallel:
                for trajectory in tqdm(original_trajectories):
                    new_trajectories = _add_new_unwrapped_state_trajectories(angle_index_combination, trajectory)
                    updated_trajectories.extend(new_trajectories)
            else:
                with mp.Pool(mp.cpu_count()) as pool:
                    new_trajectories_lists = list(
                        tqdm(
                            pool.imap(
                                partial(_add_new_unwrapped_state_trajectories, angle_index_combination), 
                                original_trajectories
                            ),
                            total=len(original_trajectories),
                        )
                    )
                    for new_trajectories in new_trajectories_lists:
                        updated_trajectories.extend(new_trajectories)
    
    if is_dict_input:
        new_trajectories, new_plans = zip(*updated_pairs) if updated_pairs else ([], [])
        return {
            "trajectories": list(new_trajectories),
            "plans": list(new_plans)
        }
    else:
        return updated_trajectories

def _unwrap_trajectory_angles(angle_indices: List[int], trajectory: np.ndarray) -> np.ndarray:
    angles = trajectory[:, angle_indices]

    # Compute the difference between successive angle values
    angle_diffs = np.diff(angles, axis = 0)

    # Determine corrections: if the difference is too large, correct by ±2π
    corrections = np.zeros_like(angles)
    corrections[1:] = np.where(angle_diffs >= np.pi, -2 * np.pi, np.where(angle_diffs <= -np.pi, 2 * np.pi, 0))

    # Compute the cumulative correction for each state
    corrections = np.cumsum(corrections, axis=0)

    # Apply the corrections to the original angles
    trajectory_copy = trajectory.copy()
    trajectory_copy[:, angle_indices] += corrections

    return trajectory_copy

def handle_angle_wraparound(data_or_trajectories, parallel=True, angle_indices: List[int] = [0], **kwargs):
    """
    Handle angle wraparound by detecting and correcting jump discontinuities.
    
    Args:
        data_or_trajectories: Either a dictionary with 'trajectories' and 'plans' keys, 
                             or a list of trajectory arrays
        parallel: Boolean flag to use parallel processing
        angle_indices: List of indices corresponding to angles
        
    Returns:
        Updated data dictionary or list of trajectories
    """
    print(f"[ utils/data_processing ] Handling angle wraparound")
    
    is_dict_input = isinstance(data_or_trajectories, dict)
    
    if is_dict_input:
        trajectories = data_or_trajectories["trajectories"]
        plans = data_or_trajectories["plans"]
    else:
        trajectories = data_or_trajectories
    
    if not parallel:
        updated_trajectories = []
        for trajectory in tqdm(trajectories):
            updated_trajectories.append(_unwrap_trajectory_angles(angle_indices, trajectory))
    else:
        with mp.Pool(mp.cpu_count()) as pool:
            updated_trajectories = list(
                tqdm(
                    pool.imap(partial(_unwrap_trajectory_angles, angle_indices), trajectories),
                    total=len(trajectories),
                )
            )
    
    if is_dict_input:
        return {
            "trajectories": updated_trajectories,
            "plans": plans
        }
    else:
        return updated_trajectories

def _convert_single_trajectory_to_signed_range(angle_indices, trajectory):
    """
    Convert angles from [0, 2π] range to [-π, π] range for a single trajectory.
    
    Args:
        angle_indices: List of indices where angles need to be converted
        trajectory: A single trajectory array
        
    Returns:
        Converted trajectory with angles in [-π, π] range
    """
    
    # Create a copy to avoid modifying the original
    converted_trajectory = trajectory.copy()
    for idx in angle_indices:
        # Convert angles > π to equivalent negative angles
        mask = converted_trajectory[..., idx] > np.pi
        converted_trajectory[..., idx][mask] -= 2 * np.pi
    return converted_trajectory

def convert_angles_to_signed_range(data_or_trajectories, parallel=True, angle_indices=[0], **kwargs):
    """
    Convert angles from [0, 2π] range to [-π, π] range at specified indices.
    
    Args:
        data_or_trajectories: Either a dictionary with 'trajectories' and 'plans' keys, 
                             or a list of trajectory arrays
        parallel: Boolean flag to use parallel processing
        angle_indices: List of indices where angles need to be converted
        
    Returns:
        Updated data dictionary or list of trajectories
    """
    print(f"[ utils/data_processing ] Converting angles to signed range")
    
    is_dict_input = isinstance(data_or_trajectories, dict)
    
    if is_dict_input:
        trajectories = data_or_trajectories["trajectories"]
        plans = data_or_trajectories["plans"]
    else:
        trajectories = data_or_trajectories
    
    if not parallel:
        converted_trajectories = []
        for trajectory in tqdm(trajectories):
            converted_trajectories.append(_convert_single_trajectory_to_signed_range(angle_indices, trajectory))
    else:
        with mp.Pool(mp.cpu_count()) as pool:
            convert_func = partial(_convert_single_trajectory_to_signed_range, angle_indices)
            converted_trajectories = list(
                tqdm(
                    pool.imap(
                        convert_func, 
                        trajectories
                    ),
                    total=len(trajectories),
                )
            )
    
    if is_dict_input:
        return {
            "trajectories": converted_trajectories,
            "plans": plans
        }
    else:
        return converted_trajectories



def shift_to_zero_center_angles(
    data_or_trajectories: Union[np.ndarray, torch.Tensor],
    *,
    angle_indices: Sequence[int],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Wrap angles in the given columns to the range (-π, π].

    Works for arrays/tensors of shape (..., D) where D includes angle columns.
    Vectorized across all indices in `angle_indices`.
    """
    if angle_indices is None or len(angle_indices) == 0:
        raise ValueError("angle_indices must be a non-empty sequence")

    # NumPy path
    if isinstance(data_or_trajectories, np.ndarray):
        out = data_or_trajectories.copy()
        two_pi = 2.0 * np.pi
        # Vectorized select of all angle columns
        a = out[..., angle_indices]
        a = np.remainder(a + np.pi, two_pi) - np.pi  # (-π, π]
        out[..., angle_indices] = a
        return out

    # PyTorch path
    elif torch.is_tensor(data_or_trajectories):
        x = data_or_trajectories.clone()
        # ensure indices are a plain list/tuple for indexing
        idx = list(angle_indices)
        pi = x.new_tensor(math.pi)
        two_pi = 2 * pi

        a = x[..., idx]                               # shape (..., K)
        a = torch.remainder(a + pi, two_pi) - pi      # (-π, π]
        x[..., idx] = a                                # safe assignment on a slice
        return x

    else:
        raise ValueError(f"Unsupported data type: {type(data_or_trajectories)}")

def compute_actual_length(length: int, stride: int) -> int:
    """Compute actual sequence length accounting for stride.

    Given a number of sampled points and a stride, computes the actual
    number of timesteps covered.

    Args:
        length: Number of sampled points (history_length or horizon_length)
        stride: Temporal spacing between sampled points

    Returns:
        Actual number of timesteps: 1 + (length - 1) * stride

    Examples:
        >>> compute_actual_length(4, 1)  # 4 points, stride 1
        4
        >>> compute_actual_length(4, 2)  # 4 points, stride 2 -> covers 7 timesteps
        7
        >>> compute_actual_length(1, 5)  # 1 point, any stride -> always 1
        1
    """
    return 1 + (length - 1) * stride


def compute_num_inference_steps(
    max_path_length: int,
    history_length: int,
    horizon_length: int,
    stride: int,
) -> int:
    """Compute number of autoregressive inference steps needed.

    Given a target path length and model parameters, computes how many
    autoregressive generation steps are needed.

    Args:
        max_path_length: Target total trajectory length in timesteps
        history_length: Number of history points the model takes as input
        horizon_length: Number of horizon points the model predicts
        stride: Temporal spacing between sampled points

    Returns:
        Number of inference steps needed to reach max_path_length

    Examples:
        >>> compute_num_inference_steps(100, 4, 8, 1)  # (100 - 4) / 8 = 12
        12
        >>> compute_num_inference_steps(50, 4, 8, 2)   # actual_hist=7, actual_horz=15
        3
    """
    actual_history = compute_actual_length(history_length, stride)
    actual_horizon = compute_actual_length(horizon_length, stride)
    actual_horizon = max(actual_horizon, 1)  # Ensure at least 1 step per iteration

    remaining = max(0, max_path_length - actual_history)
    num_steps = int(np.ceil(remaining / actual_horizon))

    # Ensure at least 1 step
    return max(num_steps, 1)


def compute_max_path_length(
    num_inference_steps: int,
    history_length: int,
    horizon_length: int,
    stride: int,
) -> int:
    """Compute max path length from number of inference steps.

    Inverse of compute_num_inference_steps. Given the number of autoregressive
    steps, computes the resulting trajectory length.

    Args:
        num_inference_steps: Number of autoregressive generation steps
        history_length: Number of history points the model takes as input
        horizon_length: Number of horizon points the model predicts
        stride: Temporal spacing between sampled points

    Returns:
        Total trajectory length in timesteps

    Examples:
        >>> compute_max_path_length(12, 4, 8, 1)  # 4 + 12*8 = 100
        100
        >>> compute_max_path_length(3, 4, 8, 2)   # actual_hist=7, actual_horz=15 -> 7 + 3*15 = 52
        52
    """
    actual_history = compute_actual_length(history_length, stride)
    actual_horizon = compute_actual_length(horizon_length, stride)

    return actual_history + num_inference_steps * actual_horizon


def warn_stride_horizon_length(horizon_length, stride, context=""):
    """
    Warn if horizon_length=1 with stride>1, since stride has no effect in this case.

    When horizon_length=1, actual_horizon = 1 + (1-1)*stride = 1 regardless of stride.
    This means each inference step only advances by 1 timestep, which may be unexpected.

    Args:
        horizon_length: Number of horizon points
        stride: Temporal spacing between sampled points
        context: Optional string describing where this warning originates (e.g., "TrajectoryDataset")
    """
    import warnings
    if horizon_length == 1 and stride > 1:
        ctx = f" in {context}" if context else ""
        warnings.warn(
            f"horizon_length=1 with stride={stride}{ctx}: stride has no effect on horizon "
            f"when horizon_length=1. Each inference step will advance by only 1 timestep. "
            f"Consider increasing horizon_length if you want stride to affect prediction steps.",
            UserWarning,
            stacklevel=3,
        )

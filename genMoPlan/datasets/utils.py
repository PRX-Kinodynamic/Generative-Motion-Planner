from collections import defaultdict, namedtuple
import numpy as np
import torch
from tqdm import tqdm

from genMoPlan.utils import parallelize_toggle, compute_actual_length


NONE_TENSOR = torch.zeros(0, dtype=torch.float32, device='cpu')
EMPTY_DICT = {}

Index = namedtuple("Index", "path_ind history_start history_end horizon_start horizon_end")
DataSample = namedtuple(
    "DataSample",
    "trajectory conditions global_query local_query mask",
    defaults=(EMPTY_DICT, NONE_TENSOR, NONE_TENSOR, NONE_TENSOR),
)
FinalStateDataSample = namedtuple("FinalStateDataSample", "histories final_states max_path_length", defaults=(NONE_TENSOR, NONE_TENSOR, None))


def _make_indices_for_single_trajectory(i, traj_length, actual_history_length, actual_horizon_length, stride, use_history_padding, use_horizon_padding, use_history_mask, history_padding_anywhere, history_padding_k):
    # If masking is enabled, allow shorter histories down to at least 1 actual element
    min_history_elements = 1 if (use_history_padding or use_history_mask) else actual_history_length
    min_horizon_elements = 0 if use_horizon_padding else actual_horizon_length

    if min_history_elements + min_horizon_elements > traj_length:
        return None

    # Ensure there is room for a full horizon sampled at `stride` starting
    # right after the last history element (which is at index `history_end - 1`).
    # The first horizon index is `history_end + stride - 1`, so we must reserve
    # an additional `(stride - 1)` positions beyond `min_horizon_elements`.
    if use_horizon_padding:
        # When padding is enabled, allow history to reach the end; horizon may be empty and padded.
        max_history_end = traj_length - min_horizon_elements
    else:
        max_history_end = traj_length - min_horizon_elements - (stride - 1)
    min_history_end = min_history_elements

    unique_indices = set()

    traj_indices = []
    
    for history_end in range(min_history_end, max_history_end + 1):
        horizon_start = history_end + stride - 1
        horizon_end = min(horizon_start + actual_horizon_length, traj_length)

        if use_history_mask:
            # Generate all possible provided-history counts k in [1, min(history_length, ceil(history_end/stride))]
            max_k = min((actual_history_length + stride - 1) // stride, (history_end + stride - 1) // stride)
            # Ensure at least one element available
            max_k = max(1, max_k)
            for k in range(1, max_k + 1):
                s_lower = history_end - k * stride
                if s_lower < 0:
                    # Not enough frames to have exactly k provided points
                    continue
                history_start = s_lower

                history_indices = tuple(range(history_start, history_end, stride))
                # Skip if empty (should not happen since k>=1)
                if len(history_indices) == 0:
                    continue
                horizon_indices = tuple(range(horizon_start, horizon_end, stride))

                index_key = (history_indices, horizon_indices)
                if index_key in unique_indices:
                    continue
                unique_indices.add(index_key)
                traj_indices.append(Index(i, history_start, history_end, horizon_start, horizon_end))
        else:
            # Always include the base (full-length when available) history window
            history_start = max(0, history_end - actual_history_length)
            history_indices = tuple(range(history_start, history_end, stride))
            horizon_indices = tuple(range(horizon_start, horizon_end, stride))
            index_key = (history_indices, horizon_indices)
            if index_key not in unique_indices:
                unique_indices.add(index_key)
                traj_indices.append(Index(i, history_start, history_end, horizon_start, horizon_end))

            # Optionally add additional shorter histories anywhere when padding is enabled
            if use_history_padding and history_padding_anywhere:
                # Determine maximum k (number of provided history points) allowed at this history_end
                max_k_allowed_by_history = (actual_history_length + stride - 1) // stride
                max_k_available = (history_end + stride - 1) // stride
                max_k = max(1, min(max_k_allowed_by_history, max_k_available))

                # Resolve which k values to emit based on configuration
                if isinstance(history_padding_k, str):
                    if history_padding_k == "k1":
                        k_values = [1]
                    elif history_padding_k == "all":
                        k_values = list(range(1, max_k + 1))
                    else:
                        # Unknown string -> default to k1
                        k_values = [1]
                elif isinstance(history_padding_k, int):
                    k_values = [history_padding_k]
                else:
                    # Expecting an iterable of ints
                    try:
                        k_values = list(history_padding_k)
                    except Exception:
                        k_values = [1]

                # Clamp and filter k values
                k_values = [k for k in k_values if isinstance(k, int) and 1 <= k <= max_k]
                if len(k_values) == 0:
                    k_values = [1]

                for k in k_values:
                    s_lower = history_end - k * stride
                    if s_lower < 0:
                        continue
                    history_start_k = s_lower
                    history_indices_k = tuple(range(history_start_k, history_end, stride))
                    if len(history_indices_k) == 0:
                        continue
                    horizon_indices_k = horizon_indices  # same horizon for a given history_end
                    index_key_k = (history_indices_k, horizon_indices_k)
                    if index_key_k in unique_indices:
                        continue
                    unique_indices.add(index_key_k)
                    traj_indices.append(Index(i, history_start_k, history_end, horizon_start, horizon_end))

    return traj_indices
    

def make_indices(path_lengths, history_length, use_history_padding, horizon_length, use_horizon_padding, stride, use_history_mask=False, parallel=True, history_padding_anywhere=True, history_padding_k="k1"):
    if use_history_padding and use_horizon_padding:
        raise ValueError("Cannot use both history and horizon padding")
    if use_history_padding and use_history_mask:
        raise ValueError("Cannot use both history padding and history mask")

    indices = []
    skip_count = 0
    
    actual_horizon_length = compute_actual_length(horizon_length, stride)
    actual_history_length = compute_actual_length(history_length, stride)

    print(f"[ datasets/utils ] Actual history length: {actual_history_length}, Actual horizon length: {actual_horizon_length}")

    args_list = [
        (
            i,
            traj_length,
            actual_history_length,
            actual_horizon_length,
            stride,
            use_history_padding,
            use_horizon_padding,
            use_history_mask,
            history_padding_anywhere,
            history_padding_k,
        )
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

def _create_mirror_padding(sequence, pad_length):
    """
    Create mirror-reflected padding of specified length.

    Args:
        sequence: [N, state_dim] available data
        pad_length: Number of padding positions needed

    Returns:
        padding: [pad_length, state_dim] mirror-reflected padding
    """
    if pad_length <= 0:
        return torch.zeros((0, sequence.shape[-1]), dtype=sequence.dtype, device=sequence.device)

    # Reverse the sequence
    reflected = sequence.flip(0)  # [N, state_dim] reversed

    # Tile if needed to cover pad_length
    if pad_length <= len(reflected):
        return reflected[:pad_length]
    else:
        # Need to tile: repeat reflected sequence
        num_tiles = (pad_length + len(reflected) - 1) // len(reflected)
        tiled = reflected.repeat(num_tiles, 1)  # [num_tiles * N, state_dim]
        return tiled[:pad_length]


def apply_padding(trajectory, length, pad_left=True, pad_value=None, strategy=None):
    """
    Apply padding to sequence to reach target length.

    Args:
        trajectory: [N, state_dim] available data
        length: M (desired length, M >= N)
        pad_left: If True, pad on left; else pad on right
        pad_value: Explicit value to use for padding (overrides strategy)
        strategy: Padding strategy if pad_value not provided
                  - "zeros": Pad with zeros
                  - "first": Pad with first element (default if pad_left=True)
                  - "last": Pad with last element (default if pad_left=False)
                  - "mirror": Pad with reflected sequence

    Returns:
        padded_sequence: [M, state_dim] padded data

    Backward Compatibility:
        - If pad_value is provided, it takes precedence over strategy
        - If neither provided, defaults to "first" for pad_left=True, "last" for pad_left=False
          (matching current behavior)
    """
    if len(trajectory) == length:
        return trajectory

    pad_length = length - len(trajectory)

    if pad_value is None and len(trajectory) == 0:
        # Special case: empty trajectory - can only pad with zeros
        if strategy == "zeros" or strategy is None:
            padding = torch.zeros((pad_length, trajectory.shape[-1]), dtype=trajectory.dtype, device=trajectory.device)
        else:
            raise ValueError(f"Cannot pad empty trajectory with strategy '{strategy}'. Use 'zeros' or provide pad_value.")
    elif pad_value is not None:
        # Explicit pad_value takes precedence
        padding = pad_value.repeat(pad_length, 1)
    elif strategy == "zeros":
        # Pad with zeros
        padding = torch.zeros((pad_length, trajectory.shape[-1]), dtype=trajectory.dtype, device=trajectory.device)
    elif strategy == "first":
        # Pad with first element
        padding = trajectory[0].repeat(pad_length, 1)
    elif strategy == "last":
        # Pad with last element
        padding = trajectory[-1].repeat(pad_length, 1)
    elif strategy == "mirror":
        # Pad with reflected sequence
        padding = _create_mirror_padding(trajectory, pad_length)
    else:
        # Default behavior (backward compatible): use first for left, last for right
        if pad_left:
            padding = trajectory[0].repeat(pad_length, 1)
        else:
            padding = trajectory[-1].repeat(pad_length, 1)

    if pad_left:
        return torch.cat([padding, trajectory], dim=0)
    else:
        return torch.cat([trajectory, padding], dim=0)
    
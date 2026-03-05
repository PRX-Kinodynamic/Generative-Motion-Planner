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
    "trajectory conditions global_query local_query mask rollout_targets",
    defaults=(EMPTY_DICT, NONE_TENSOR, NONE_TENSOR, NONE_TENSOR, NONE_TENSOR),
)
FinalStateDataSample = namedtuple("FinalStateDataSample", "histories final_states full_trajectories", defaults=(NONE_TENSOR, NONE_TENSOR, None))


def _build_shared_shifts_from_config(adaptive_rollout, num_transitions):
    if num_transitions <= 0:
        return []

    if adaptive_rollout is None:
        adaptive_rollout = {}

    schedule = adaptive_rollout.get("shared_shift_schedule", {"type": "arithmetic", "start": 1, "delta": 0})

    if isinstance(schedule, (list, tuple)):
        shifts = [int(v) for v in schedule]
    elif isinstance(schedule, dict):
        schedule_type = schedule.get("type", "arithmetic")
        if schedule_type == "arithmetic":
            start = int(schedule.get("start", 1))
            delta = int(schedule.get("delta", 0))
            shifts = [start + delta * k for k in range(num_transitions)]
        elif schedule_type == "explicit":
            values = schedule.get("values", [])
            shifts = [int(v) for v in values]
        else:
            raise ValueError(f"Unsupported shared_shift_schedule type: {schedule_type}")
    else:
        raise ValueError("shared_shift_schedule must be a dict, list, or tuple")

    if len(shifts) != num_transitions:
        raise ValueError(
            f"shared_shift_schedule must provide {num_transitions} transitions, got {len(shifts)}"
        )
    if any(v < 0 for v in shifts):
        raise ValueError(f"shared_shift_schedule values must be non-negative, got {shifts}")

    return shifts


def compute_required_future_for_adaptive_rollout(actual_horizon_length, stride, rollout_steps, adaptive_rollout=None):
    if rollout_steps <= 1:
        return actual_horizon_length + (stride - 1)

    if adaptive_rollout is None:
        adaptive_rollout = {}

    sampled_horizon_length = ((actual_horizon_length - 1) // stride) + 1
    base_span = int(adaptive_rollout.get("base_span", sampled_horizon_length))
    if base_span <= 0:
        raise ValueError(f"adaptive_rollout.base_span must be > 0, got {base_span}")

    raw_target_span = compute_actual_length(base_span, stride)
    num_rollout_targets = rollout_steps - 1
    num_transitions = max(0, num_rollout_targets - 1)
    shared_shifts = _build_shared_shifts_from_config(adaptive_rollout, num_transitions)
    max_offset = sum(shared_shifts)

    # horizon_start = history_end + (stride - 1)
    # last_target_start = horizon_start + max_offset * stride
    # last_target_end = last_target_start + raw_target_span
    required_future = (stride - 1) + max_offset * stride + raw_target_span
    return required_future


def _make_indices_for_single_trajectory(i, traj_length, actual_history_length, actual_horizon_length, stride, use_history_padding, use_horizon_padding, use_history_mask, history_padding_anywhere, history_padding_k, rollout_steps=1, rollout_target_mode="gt_future", adaptive_rollout=None):
    # If masking is enabled, allow shorter histories down to at least 1 actual element
    min_history_elements = 1 if (use_history_padding or use_history_mask) else actual_history_length
    min_horizon_elements = 0 if use_horizon_padding else actual_horizon_length
    
    # Account for rollout_steps: require enough future raw frames to extract (rollout_steps-1) additional
    # strided horizon chunks, plus the (stride-1) gaps between chunk starts (consistent with horizon_start).
    if use_horizon_padding:
        min_total_length = min_history_elements + min_horizon_elements
    else:
        if rollout_target_mode == "adaptive_stride" and rollout_steps > 1:
            required_future = compute_required_future_for_adaptive_rollout(
                actual_horizon_length=actual_horizon_length,
                stride=stride,
                rollout_steps=rollout_steps,
                adaptive_rollout=adaptive_rollout,
            )
        else:
            required_future = rollout_steps * actual_horizon_length + rollout_steps * (stride - 1)
        min_total_length = min_history_elements + required_future

    if min_total_length > traj_length:
        return None

    # Ensure there is room for a full horizon sampled at `stride` starting
    # right after the last history element (which is at index `history_end - 1`).
    # The first horizon index is `history_end + stride - 1`, so we must reserve
    # an additional `(stride - 1)` positions beyond `min_horizon_elements`.
    if use_horizon_padding:
        # When padding is enabled, allow history to reach the end; horizon may be empty and padded.
        max_history_end = traj_length - min_horizon_elements
    else:
        if rollout_target_mode == "adaptive_stride" and rollout_steps > 1:
            required_future = compute_required_future_for_adaptive_rollout(
                actual_horizon_length=actual_horizon_length,
                stride=stride,
                rollout_steps=rollout_steps,
                adaptive_rollout=adaptive_rollout,
            )
            max_history_end = traj_length - required_future
        else:
            # Also reserve space for future rollout targets when rollout_steps > 1.
            extra_rollout = 0
            if rollout_steps > 1:
                # For each additional horizon chunk: need its raw span plus the (stride-1) gap before it.
                extra_rollout = (rollout_steps - 1) * (actual_horizon_length + (stride - 1))
            max_history_end = traj_length - min_horizon_elements - (stride - 1) - extra_rollout
    min_history_end = min_history_elements
    if max_history_end < min_history_end:
        return None

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
    

def make_indices(path_lengths, history_length, use_history_padding, horizon_length, use_horizon_padding, stride, use_history_mask=False, parallel=True, history_padding_anywhere=True, history_padding_k="k1", rollout_steps=1, rollout_target_mode="gt_future", adaptive_rollout=None):
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
            rollout_steps,
            rollout_target_mode,
            adaptive_rollout,
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
    

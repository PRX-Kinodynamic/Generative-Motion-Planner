import numpy as np


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
        >>> compute_num_inference_steps(50, 4, 8, 2)   # actual_hist=7, advance=16
        3
    """
    actual_history = compute_actual_length(history_length, stride)
    advance_per_step = max(horizon_length * stride, 1)

    remaining = max(0, max_path_length - actual_history)
    num_steps = int(np.ceil(remaining / advance_per_step))

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
        >>> compute_max_path_length(3, 4, 8, 2)   # actual_hist=7, advance=16 -> 7 + 3*16 = 55
        55
    """
    actual_history = compute_actual_length(history_length, stride)
    advance_per_step = horizon_length * stride

    return actual_history + num_inference_steps * advance_per_step


def compute_total_predictions(
    history_length: int,
    num_inference_steps: int,
    horizon_length: int,
) -> int:
    """Compute total number of model predictions (in model-space).

    This is the number of states the model will output during generation,
    which is useful for allocating trajectory buffers efficiently.

    Args:
        history_length: Number of history points the model takes as input
        num_inference_steps: Number of autoregressive generation steps
        horizon_length: Number of horizon points the model predicts per step

    Returns:
        Total number of model predictions

    Examples:
        >>> compute_total_predictions(1, 20, 31)
        621  # 1 (history) + 20 * 31 (predictions)
        >>> compute_total_predictions(3, 15, 5)
        78  # 3 (history) + 15 * 5 (predictions)
    """
    return history_length + num_inference_steps * horizon_length

import numpy as np
from genMoPlan.utils.roa import ROAEstimator
from genMoPlan.datasets.normalization import get_normalizer, Normalizer
from genMoPlan.utils.model import get_normalizer_params


def compute_final_state_variance(final_states, angle_indices):
    """
    Compute variance per start state across runs, handling angular data correctly.
    
    Parameters:
    final_states: np.array of shape (num_start_states, num_runs, dimensions)
    angle_indices: list of indices corresponding to angular dimensions
    
    Returns:
    variance: np.array of shape (num_start_states, dimensions)
    """
    num_start_states, num_runs, dimensions = final_states.shape
    variance = np.zeros((num_start_states, dimensions))
    
    # Regular variance for non-angular dimensions
    for d in range(dimensions):
        if d not in angle_indices:
            variance[:, d] = np.var(final_states[:, :, d], axis=1)
    
    # Circular variance for angular dimensions
    for d in angle_indices:
        angles = final_states[:, :, d]
        sin_values = np.sin(angles)
        cos_values = np.cos(angles)
        
        # Calculate mean resultant vector length per start state
        mean_sin = np.mean(sin_values, axis=1)
        mean_cos = np.mean(cos_values, axis=1)
        r = np.sqrt(mean_sin**2 + mean_cos**2)
        
        # Circular variance = 1 - R
        variance[:, d] = 1 - r
    
    return variance


def compute_merged_variance_score(variance_array, model_args, normalizer_params):
    """
    Merge position and angular variances into a single score per start state.
    
    Parameters:
    variance_array: np.array of shape (num_start_states, 2)
    
    Returns:
    merged_score: np.array of shape (num_start_states,)
    """
    normalizer: Normalizer = get_normalizer(model_args.trajectory_normalizer, normalizer_params)
    # Get variables for normalization
    pos_var = variance_array[:, 0]
    angle_var = variance_array[:, 1]  # Already in [0,1] range
    
    # Use the normalizer to normalize position variance
    pos_var_reshaped = pos_var.reshape(-1, 1)  # Reshape for normalization
    norm_pos_var = normalizer.normalize(pos_var_reshaped, [0]).flatten()
    
    # Weighted sum (adjust weights based on importance)
    weights = np.array([0.5, 0.5])  # Equal weights
    merged_score = weights[0] * norm_pos_var + weights[1] * angle_var
    
    return merged_score


def evaluate_final_state_variance(roa_estimator: ROAEstimator, horizon_length: int, num_inference_steps: int, angle_indices: list, inference_normalizer_params: dict):
    """
    Evaluate the variance of the final states of the trajectories.
    
    Parameters:
    roa_estimator: ROAEstimator instance
    horizon_length: int, length of prediction horizon
    num_inference_steps: int, number of inference steps
    angle_indices: list of indices corresponding to angular dimensions
    
    Returns:
    tuple: (variance_per_state, merged_score)
        - variance_per_state: np.array of shape (num_start_states, 2)
        - merged_score: np.array of shape (num_start_states,)
    """
    roa_estimator.set_horizon_and_max_path_lengths(horizon_length, num_inference_steps)
    roa_estimator.generate_trajectories(compute_labels=False, discard_trajectories=True, save=True)

    final_states = roa_estimator.final_states
    variance_per_state = compute_final_state_variance(final_states, angle_indices)
    merged_score = compute_merged_variance_score(variance_per_state, roa_estimator.model_args, inference_normalizer_params)
    
    return variance_per_state, merged_score
    

    

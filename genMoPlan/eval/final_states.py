from collections.abc import Callable
from typing import List
import numpy as np
from tqdm import tqdm
from scipy.stats import circstd
from genMoPlan.datasets.normalization import get_normalizer, Normalizer
from genMoPlan.models.generative.base import GenerativeModel
from genMoPlan.utils.json_args import JSONArgs


def compute_final_state_std(final_states, angle_indices):
    """
    Compute variance per start state across runs, handling angular data correctly.
    
    Parameters:
    final_states: np.array of shape (num_start_states, num_runs, dimensions)
    angle_indices: list of indices corresponding to angular dimensions
    
    Returns:
    variance: np.array of shape (num_start_states, dimensions)
    """
    num_start_states, num_runs, dimensions = final_states.shape
    non_angular_indices = [i for i in range(dimensions) if i not in angle_indices]

    std = np.zeros((num_start_states, dimensions))

    std[:, non_angular_indices] = np.std(final_states[:, :, non_angular_indices], axis=1)
    std[:, angle_indices] = circstd(final_states[:, :, angle_indices], high=np.pi, low=-np.pi, axis=1)
    
    return std

def evaluate_final_state_std(
        model: GenerativeModel, 
        start_states: np.ndarray, 
        model_args: JSONArgs, 
        n_runs: int,
        num_inference_steps: int, 
        inference_normalization_params: dict, 
        device: str,
        batch_size: int = 5000,
        conditional_sample_kwargs: dict = {},
        post_process_fns: List[Callable] = [],
        post_process_fn_kwargs: dict = {},
    ):
    from genMoPlan.utils import generate_trajectories

    max_path_length = (num_inference_steps * model_args.horizon_length) + model_args.history_length

    num_start_states, dim = start_states.shape

    final_states = np.zeros((num_start_states, n_runs, dim))

    for i in tqdm(range(n_runs), desc="Generating trajectories for uncertainty computation"):
        run_final_states = generate_trajectories(
            model, 
            model_args, 
            start_states, 
            max_path_length, 
            device,
            verbose=True,
            batch_size=batch_size,
            conditional_sample_kwargs=conditional_sample_kwargs,
            only_return_final_states=True,
            post_process_fns=post_process_fns,
            post_process_fn_kwargs=post_process_fn_kwargs,
            horizon_length=model_args.horizon_length,
        )
        normalizer: Normalizer = get_normalizer(model_args.trajectory_normalizer, inference_normalization_params)
        final_states[:, i, :] = normalizer(run_final_states)

    std_per_state = compute_final_state_std(final_states, model_args.angle_indices)
    merged_std = np.mean(std_per_state, axis=1)
    
    return merged_std
    


def plot_final_state_std(std, start_states, save_path: str, title: str, s=1):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    plt.figure(figsize=(10, 8))
    
    # Apply log normalization
    scatter = plt.scatter(start_states[:, 0], start_states[:, 1], 
                      c=std, cmap=cm.viridis,
                      alpha=1, edgecolors='none', s=s)

    plt.colorbar(scatter, label='Std')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close()
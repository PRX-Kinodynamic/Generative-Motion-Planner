import numpy as np
from scipy.stats import circstd
from genMoPlan.utils.roa import ROAEstimator
from genMoPlan.datasets.normalization import get_normalizer, Normalizer


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


def evaluate_final_state_std(roa_estimator: ROAEstimator, horizon_length: int, num_inference_steps: int, angle_indices: list, inference_normalization_params: dict):
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
    roa_estimator.set_horizon_and_max_path_lengths(horizon_length, num_inference_steps=num_inference_steps)
    roa_estimator.generate_trajectories(compute_labels=False, discard_trajectories=True, save=False)

    final_states = roa_estimator.final_states

    normalizer: Normalizer = get_normalizer(roa_estimator.model_args.trajectory_normalizer, inference_normalization_params)
    norm_final_states = normalizer(final_states)

    std_per_state = compute_final_state_std(norm_final_states, angle_indices)
    merged_std = np.mean(std_per_state, axis=1)
    
    return merged_std


def plot_final_state_std(std, start_states, save_path: str, title: str):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    plt.figure(figsize=(10, 8))
    
    # Apply log normalization
    scatter = plt.scatter(start_states[:, 0], start_states[:, 1], 
                      c=std, cmap=cm.viridis,
                      alpha=1, edgecolors='none', s=1)

    plt.colorbar(scatter, label='Std')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_final_state_log_std(std, start_states, save_path: str, title: str):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    log_std = np.log(np.log(np.log(std + 1) + 1) + 1)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(start_states[:, 0], start_states[:, 1], 
                      c=log_std, cmap=cm.viridis,
                      alpha=1, edgecolors='none', s=1)

    plt.colorbar(scatter, label='Log Std')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_final_state_eight_root_std(std, start_states, save_path: str, title: str):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    eight_root_std = np.power(std, 1/8)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(start_states[:, 0], start_states[:, 1], 
                        c=eight_root_std, cmap=cm.viridis, 
                        alpha=1, edgecolors='none', s=1)

    plt.colorbar(scatter, label='Eight Root Std')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_final_state_std_sigmoid(std, start_states, save_path: str, title: str):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    sigmoid_std = 1 / (1 + np.exp(-std))

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(start_states[:, 0], start_states[:, 1], 
                        c=sigmoid_std, cmap=cm.viridis, 
                        alpha=1, edgecolors='none', s=1)

    plt.colorbar(scatter, label='Sigmoid Std')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close()
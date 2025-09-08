from abc import ABC, abstractmethod
from collections.abc import Callable
import os
from typing import List

import numpy as np
from tqdm import tqdm
from scipy.stats import circstd, circvar
import matplotlib.pyplot as plt
from matplotlib import cm

from genMoPlan.datasets.normalization import get_normalizer, Normalizer
from genMoPlan.utils import JSONArgs, get_non_angular_indices


def _normalize_final_states(
    unnormalized_final_states: np.ndarray,
    model_args: JSONArgs, 
    inference_normalization_params: dict, 
):
    """Normalize final states to [-1, 1] for a uniform uncertainty computation across all dimensions."""

    n_runs = unnormalized_final_states.shape[1]

    final_states = np.zeros_like(unnormalized_final_states)

    # TODO: Can be optimized by running the normalizer on the whole array at once
    for i in tqdm(range(n_runs), desc="Normalizing final states"):
        run_final_states = unnormalized_final_states[:, i, :]

        # Normalize the final states to [-1, 1] for a uniform uncertainty computation across all dimensions
        normalizer: Normalizer = get_normalizer(model_args.trajectory_normalizer, inference_normalization_params)
        final_states[:, i, :] = normalizer(run_final_states)

    
    return final_states


class Uncertainty(ABC):
    """Base class for uncertainty estimators.
    Any uncertainty estimator should inherit from this class and implement the following methods:
    - _compute_non_angular_uncertainty
    - _compute_angular_uncertainty

    The uncertainty is computed by averaging over all dimensions and then normalized to [0, 1].

    The normalized uncertainty is saved to a file and plotted.
    """

    name: str

    def __init__(
        self,
        inference_normalization_params: dict = None,
    ):
        self.inference_normalization_params = inference_normalization_params

    @abstractmethod
    def _compute_non_angular_uncertainty(self, final_states: np.ndarray, non_angular_indices: List[int]) -> np.ndarray: ...
    
    @abstractmethod
    def _compute_angular_uncertainty(self, final_states: np.ndarray, angle_indices: List[int]) -> np.ndarray: ...

    def compute_normalized_uncertainty(
        self, 
        unnormalized_final_states: np.ndarray, 
        model_args: JSONArgs, 
        start_states: np.ndarray, 
        save_path: str, 
        title_suffix: str,
    ) -> np.ndarray:

        final_states = _normalize_final_states(
            unnormalized_final_states,
            model_args,
            self.inference_normalization_params,
        )

        angle_indices = model_args.angle_indices

        num_start_states, num_runs, dimensions = final_states.shape
        non_angular_indices = get_non_angular_indices(angle_indices, dimensions)

        uncertainty_per_dim = np.zeros((num_start_states, dimensions))

        uncertainty_per_dim[:, non_angular_indices] = self._compute_non_angular_uncertainty(final_states, non_angular_indices)

        uncertainty_per_dim[:, angle_indices] = self._compute_angular_uncertainty(final_states, angle_indices)

        # Average over all dimensions
        uncertainty = np.mean(uncertainty_per_dim, axis=1)

        # Normalize the uncertainty to [0, 1]
        normalized_uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-8)

        np.save(os.path.join(save_path, "normalized_uncertainty.npy"), normalized_uncertainty)

        self.plot_uncertainty(normalized_uncertainty, start_states, save_path, title_suffix)

        return normalized_uncertainty

    def plot_uncertainty(
        self, 
        uncertainty: np.ndarray, 
        start_states: np.ndarray, 
        save_path: str, 
        title_suffix: str,
    ):
        plot_path = os.path.join(save_path, "normalized_uncertainty.png")
        title = f"{self.name} - {title_suffix}"
        plt.figure(figsize=(10.08, 8))
    
        scatter = plt.scatter(
            start_states[:, 0], start_states[:, 1], 
            c=uncertainty, 
            cmap=cm.viridis,
            alpha=1, 
            edgecolors='none', 
            s=10,
        )

        plt.colorbar(scatter, label='Normalized Uncertainty')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        plt.savefig(plot_path, dpi=300)
        plt.close()

class FinalStateStd(Uncertainty):
    name: str = "Normalized Std Dev"

    def _compute_non_angular_uncertainty(self, final_states: np.ndarray, non_angular_indices: List[int]) -> np.ndarray:
        return np.std(final_states[:, :, non_angular_indices], axis=1)

    def _compute_angular_uncertainty(self, final_states: np.ndarray, angle_indices: List[int]) -> np.ndarray:
        return circstd(final_states[:, :, angle_indices], high=np.pi, low=-np.pi, axis=1)

class FinalStateVariance(Uncertainty):
    name: str = "Normalized Variance"

    def _compute_non_angular_uncertainty(self, final_states: np.ndarray, non_angular_indices: List[int]) -> np.ndarray:
        return np.var(final_states[:, :, non_angular_indices], axis=1)

    def _compute_angular_uncertainty(self, final_states: np.ndarray, angle_indices: List[int]) -> np.ndarray:
        return circvar(final_states[:, :, angle_indices], high=np.pi, low=-np.pi, axis=1)

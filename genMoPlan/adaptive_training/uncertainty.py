from abc import ABC, abstractmethod
from collections.abc import Callable
import os
from typing import List

import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import circstd, circvar
import matplotlib.pyplot as plt
from matplotlib import cm

from genMoPlan.datasets.normalization import get_normalizer, Normalizer
from genMoPlan.models.generative.base import GenerativeModel
from genMoPlan.utils import JSONArgs, _get_non_angular_indices
from genMoPlan.utils.trajectory import generate_trajectories


def _generate_final_states_for_uncertainty(
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
    """Generate final states for uncertainty computation and normalize them to [-1, 1] for a uniform uncertainty computation across all dimensions."""

    model.eval()

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

        # Normalize the final states to [-1, 1] for a uniform uncertainty computation across all dimensions
        normalizer: Normalizer = get_normalizer(model_args.trajectory_normalizer, inference_normalization_params)
        final_states[:, i, :] = normalizer(run_final_states)

    
    return final_states


class Uncertainty(ABC):
    """Base class for uncertainty estimators.
    Any uncertainty estimator should inherit from this class and implement the following methods:
    - _compute_non_angular_uncertainty
    - _compute_angular_uncertainty

    The uncertainty is computed by averaging over all dimensions.

    The uncertainty is saved to a file and plotted.
    """

    name: str

    def __init__(
        self,
        n_runs: int,
        device: str,
        angle_indices: List[int],
        batch_size: int = int(1e6),
        inference_normalization_params: dict = None,
        conditional_sample_kwargs: dict = {},
        post_process_fns: List[Callable] = [],
        post_process_fn_kwargs: dict = {},
        num_inference_steps: int = None,
    ):
        self.n_runs = n_runs
        self.batch_size = batch_size
        self.angle_indices = angle_indices
        self.device = device
        self.inference_normalization_params = inference_normalization_params
        self.conditional_sample_kwargs = conditional_sample_kwargs
        self.post_process_fns = post_process_fns
        self.post_process_fn_kwargs = post_process_fn_kwargs
        self.num_inference_steps = num_inference_steps

    @abstractmethod
    def _compute_non_angular_uncertainty(self, final_states: np.ndarray, non_angular_indices: List[int]) -> np.ndarray: ...
    
    @abstractmethod
    def _compute_angular_uncertainty(self, final_states: np.ndarray, angle_indices: List[int]) -> np.ndarray: ...

    def compute(
        self, 
        model: GenerativeModel, 
        model_args: JSONArgs, 
        start_states: np.ndarray, 
        save_path: str, 
        title_suffix: str,
    ) -> np.ndarray: 
        final_states = _generate_final_states_for_uncertainty(
            model,
            start_states,
            model_args,
            self.n_runs,
            self.num_inference_steps,
            self.inference_normalization_params,
            self.device,
            self.batch_size,
            self.conditional_sample_kwargs,
            self.post_process_fns,
            self.post_process_fn_kwargs,
        )

        angle_indices = model_args.angle_indices

        num_start_states, num_runs, dimensions = final_states.shape
        non_angular_indices = _get_non_angular_indices(angle_indices, dimensions)

        uncertainty_per_dim = np.zeros((num_start_states, dimensions))

        uncertainty_per_dim[:, non_angular_indices] = self._compute_non_angular_uncertainty(final_states, non_angular_indices)

        uncertainty_per_dim[:, angle_indices] = self._compute_angular_uncertainty(final_states, angle_indices)

        # Average over all dimensions
        uncertainty = np.mean(uncertainty_per_dim, axis=1)

        np.save(os.path.join(save_path, "uncertainty.npy"), uncertainty)

        self.plot_uncertainty(uncertainty, start_states, save_path, title_suffix)

        return uncertainty

    def plot_uncertainty(
        self, 
        uncertainty: np.ndarray, 
        start_states: np.ndarray, 
        save_path: str, 
        title_suffix: str,
    ):
        plot_path = os.path.join(save_path, "uncertainty.png")
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

        plt.colorbar(scatter, label='Uncertainty')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        plt.savefig(plot_path, dpi=300)
        plt.close()

class FinalStateStd(Uncertainty):
    name: str = "Final State Standard Deviation"

    def _compute_non_angular_uncertainty(self, final_states: np.ndarray, non_angular_indices: List[int]) -> np.ndarray:
        return np.std(final_states[:, :, non_angular_indices], axis=1)

    def _compute_angular_uncertainty(self, final_states: np.ndarray, angle_indices: List[int]) -> np.ndarray:
        return circstd(final_states[:, :, angle_indices], high=np.pi, low=-np.pi, axis=1)

class FinalStateVariance(Uncertainty):
    name: str = "Final State Variance"

    def _compute_non_angular_uncertainty(self, final_states: np.ndarray, non_angular_indices: List[int]) -> np.ndarray:
        return np.var(final_states[:, :, non_angular_indices], axis=1)

    def _compute_angular_uncertainty(self, final_states: np.ndarray, angle_indices: List[int]) -> np.ndarray:
        return circvar(final_states[:, :, angle_indices], high=np.pi, low=-np.pi, axis=1)

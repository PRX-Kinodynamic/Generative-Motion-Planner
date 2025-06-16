from collections.abc import Callable
import os
import numpy as np
from typing import List, Union
import torch
from genMoPlan.models.generative.base import GenerativeModel
from genMoPlan.eval.final_states import compute_final_state_std, evaluate_final_state_std, plot_final_state_std
from genMoPlan.utils import JSONArgs


class Uncertainty:
    """Base uncertainty estimator."""
    def compute(self, model: GenerativeModel, start_states: np.ndarray, savepath: str) -> np.ndarray:
        raise NotImplementedError


class FinalStateStd(Uncertainty):
    """
    Estimate uncertainty by the std of final states over multiple rollout runs.
    """

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

    @torch.no_grad()
    def compute(self, model: GenerativeModel, model_args: JSONArgs, start_states: np.ndarray, save_path: str, title_suffix: str) -> np.ndarray:
        model.eval()

        std = evaluate_final_state_std(
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

        np.save(os.path.join(save_path, "uncertainty.npy"), std)

        plot_final_state_std(std, start_states, save_path=os.path.join(save_path, "uncertainty.png"), title=f"Final State Standard Deviation - {title_suffix}", s=10)

        return std

import os
import numpy as np
from typing import List, Optional
import torch
from genMoPlan.models.generative.base import GenerativeModel
from genMoPlan.eval.final_states import compute_final_state_std, plot_final_state_std


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
        n_runs: int = 10,
        batch_size: int = 256,
        angle_indices: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        self.n_runs = n_runs
        self.batch_size = batch_size
        self.angle_indices = angle_indices or []
        self.device = device

    @torch.no_grad()
    def compute(self, model: GenerativeModel, start_states: np.ndarray, savepath: str, titleSuffix: str) -> np.ndarray:
        model.eval()

        n_states, dim = start_states.shape
        final_states = np.zeros((n_states, self.n_runs, dim), dtype=np.float32)

        for run in range(self.n_runs):
            for idx in range(0, n_states, self.batch_size):
                batch_slice = slice(idx, min(idx + self.batch_size, n_states))
                batch_states = start_states[batch_slice]

                batch_tensor = torch.from_numpy(batch_states).float().to(self.device)

                # history length assumed 1
                cond = {0: batch_tensor}

                sample = model.forward(cond=cond, global_query=None, local_query=None, verbose=False)

                batch_final = sample.trajectories[:, -1, :].cpu().numpy()
                final_states[batch_slice, run, :] = batch_final

        std_per_state = compute_final_state_std(final_states, self.angle_indices)
        merged_std = np.mean(std_per_state, axis=1)

        np.save(os.path.join(savepath, "uncertainty.npy"), merged_std)

        plot_final_state_std(merged_std, start_states, savepath=savepath, title=f"Final State Std - {titleSuffix}")

        return merged_std

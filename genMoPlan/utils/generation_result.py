"""Data structures for trajectory generation results."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["GenerationResult", "TerminationResult"]


@dataclass
class TerminationResult:
    """Result of termination checking for a batch of trajectories.

    Attributes:
        step: [batch] step index where each trajectory terminated, -1 if not terminated
        outcome: [batch] Outcome enum value at termination, -1 if not terminated
        any_terminated: True if any trajectory in the batch terminated
        all_terminated: True if all trajectories in the batch terminated
    """

    step: np.ndarray
    outcome: np.ndarray
    any_terminated: bool
    all_terminated: bool

    @classmethod
    def none(cls, batch_size: int) -> "TerminationResult":
        """Create result indicating no termination for any trajectory."""
        return cls(
            step=np.full(batch_size, -1, dtype=np.int32),
            outcome=np.full(batch_size, -1, dtype=np.int32),
            any_terminated=False,
            all_terminated=False,
        )

    def update(self, new_terminations: "TerminationResult") -> "TerminationResult":
        """Update this result with newly terminated trajectories.

        Only updates trajectories that haven't already terminated.
        """
        # Find trajectories that just terminated (weren't terminated before, are now)
        newly_terminated = (self.step < 0) & (new_terminations.step >= 0)

        # Update step and outcome for newly terminated trajectories
        updated_step = self.step.copy()
        updated_outcome = self.outcome.copy()
        updated_step[newly_terminated] = new_terminations.step[newly_terminated]
        updated_outcome[newly_terminated] = new_terminations.outcome[newly_terminated]

        return TerminationResult(
            step=updated_step,
            outcome=updated_outcome,
            any_terminated=(updated_step >= 0).any(),
            all_terminated=(updated_step >= 0).all(),
        )


@dataclass
class GenerationResult:
    """Result of trajectory generation.

    Attributes:
        final_states: [batch, state_dim] final state for each trajectory
        trajectories: [batch, path_length, state_dim] full trajectories, None if not requested
        termination_steps: [batch] step index where each trajectory terminated, -1 if not terminated
        termination_outcomes: [batch] Outcome enum value at termination, -1 if not terminated
    """

    final_states: np.ndarray
    trajectories: Optional[np.ndarray] = None
    termination_steps: Optional[np.ndarray] = None
    termination_outcomes: Optional[np.ndarray] = None

    @property
    def batch_size(self) -> int:
        """Number of trajectories in this result."""
        return self.final_states.shape[0]

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state space."""
        return self.final_states.shape[-1]

    @property
    def has_trajectories(self) -> bool:
        """Whether full trajectories are available."""
        return self.trajectories is not None

    @property
    def has_termination_info(self) -> bool:
        """Whether termination information is available."""
        return self.termination_steps is not None

    def terminated_mask(self) -> Optional[np.ndarray]:
        """Boolean mask of trajectories that terminated early."""
        if self.termination_steps is None:
            return None
        return self.termination_steps >= 0

    def num_terminated(self) -> int:
        """Number of trajectories that terminated early."""
        mask = self.terminated_mask()
        if mask is None:
            return 0
        return int(mask.sum())

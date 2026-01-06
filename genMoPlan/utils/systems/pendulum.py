from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from genMoPlan.datasets.normalization import Normalizer
from genMoPlan.utils.systems.base import BaseSystem, Outcome
from genMoPlan.utils.data_processing import (
    handle_angle_wraparound,
    augment_unwrapped_state_data,
)


class PendulumLQRSystem(BaseSystem):
    """
    System descriptor for Pendulum with LQR control.

    State: [theta, theta_dot]
    - theta: pendulum angle (wrapped, on FlatTorus manifold)
    - theta_dot: angular velocity

    The goal is to swing up and stabilize at the upright position (theta=0).
    """

    # Class-level defaults for state space
    DEFAULT_STATE_DIM = 2
    DEFAULT_MINS = [-2 * np.pi, -2 * np.pi]
    DEFAULT_MAXS = [2 * np.pi, 2 * np.pi]
    DEFAULT_MAX_PATH_LENGTH = 502
    DEFAULT_ANGLE_INDICES = [0]
    DEFAULT_STATE_NAMES = ["theta", "theta_dot"]

    def __init__(
        self,
        *,
        name: str = "pendulum_lqr",
        state_dim: int = 2,
        stride: int,
        history_length: int,
        horizon_length: int,
        max_path_length: Optional[int] = None,
        mins: Optional[List[float]] = None,
        maxs: Optional[List[float]] = None,
        state_names: Optional[List[str]] = None,
        angle_indices: Optional[List[int]] = None,
        valid_outcomes: Optional[Sequence[Outcome]] = None,
        normalizer: Optional[Normalizer] = None,
        metadata: Optional[Dict[str, Any]] = None,
        success_threshold: float = 0.075,
    ):
        # Set defaults from class constants
        if max_path_length is None:
            max_path_length = self.DEFAULT_MAX_PATH_LENGTH
        if mins is None:
            mins = self.DEFAULT_MINS.copy()
        if maxs is None:
            maxs = self.DEFAULT_MAXS.copy()
        if state_names is None:
            state_names = self.DEFAULT_STATE_NAMES.copy()
        if angle_indices is None:
            angle_indices = self.DEFAULT_ANGLE_INDICES.copy()
        if valid_outcomes is None:
            # Pendulum trajectories are ultimately classified as success or failure;
            # INVALID can be used by higher-level analysis when labels are uncertain.
            valid_outcomes = [Outcome.SUCCESS, Outcome.FAILURE, Outcome.INVALID]

        # Set up metadata
        metadata = metadata or {}
        metadata.setdefault("success_threshold", success_threshold)
        metadata.setdefault("angle_indices", angle_indices)

        # Preprocessing functions for Euclidean flow matching / diffusion
        trajectory_preprocess_fns = [
            handle_angle_wraparound,
            augment_unwrapped_state_data,
        ]
        preprocess_kwargs = {
            "trajectory": {
                "angle_indices": angle_indices,
            },
            "plan": None,
        }

        # No special post-processing for pendulum (angles stay in natural range)
        post_process_fns = []
        post_process_fn_kwargs = {}

        super().__init__(
            name=name,
            state_dim=state_dim,
            stride=stride,
            history_length=history_length,
            horizon_length=horizon_length,
            max_path_length=max_path_length,
            mins=mins,
            maxs=maxs,
            state_names=state_names,
            angle_indices=angle_indices,
            trajectory_preprocess_fns=trajectory_preprocess_fns,
            preprocess_kwargs=preprocess_kwargs,
            post_process_fns=post_process_fns,
            post_process_fn_kwargs=post_process_fn_kwargs,
            valid_outcomes=valid_outcomes,
            normalizer=normalizer,
            metadata=metadata,
        )

    def _wrap_angle(self, theta: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return ((theta + np.pi) % (2 * np.pi)) - np.pi

    def evaluate_final_state(self, state: np.ndarray) -> Outcome:
        """
        Evaluate the final state of a trajectory.

        Success if pendulum is near upright (theta ~= 0) with low velocity.
        """
        threshold = self.metadata.get("success_threshold", 0.075)

        theta = self._wrap_angle(state[0])
        theta_dot = state[1]

        # Check distance to upright position
        upright_distance = np.sqrt(theta**2 + theta_dot**2)

        if upright_distance <= threshold:
            return Outcome.SUCCESS

        return Outcome.FAILURE

    def should_terminate(
        self, state: np.ndarray, t: int, traj_so_far: Optional[np.ndarray]
    ):
        """
        Early termination check.

        Terminate early if pendulum reaches upright position and stabilizes.
        """
        threshold = self.metadata.get("success_threshold", 0.075)

        theta = self._wrap_angle(state[0])
        theta_dot = state[1]

        upright_distance = np.sqrt(theta**2 + theta_dot**2)

        if upright_distance <= threshold:
            return Outcome.SUCCESS

        return None

    @classmethod
    def create(
        cls,
        stride: int = 1,
        history_length: int = 1,
        horizon_length: int = 31,
        max_path_length: Optional[int] = None,
        **kwargs,
    ) -> "PendulumLQRSystem":
        """
        Factory method to create a PendulumLQRSystem with sensible defaults.

        Args:
            stride: Stride for trajectory sampling
            history_length: Length of history conditioning
            horizon_length: Length of prediction horizon
            max_path_length: Maximum trajectory length (uses default if None)
            **kwargs: Additional arguments passed to __init__

        Returns:
            PendulumLQRSystem instance
        """
        return cls(
            stride=stride,
            history_length=history_length,
            horizon_length=horizon_length,
            max_path_length=max_path_length,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        dataset_size: Optional[str] = None,
        **kwargs,
    ) -> "PendulumLQRSystem":
        """
        Create a PendulumLQRSystem from a config dictionary.

        This method extracts training parameters (stride, history_length, etc.)
        from the config while using system defaults for state space properties.

        Args:
            config: Configuration dictionary
            dataset_size: Optional dataset size indicator (e.g., "5k" or "50k")
            **kwargs: Additional arguments to override config values

        Returns:
            PendulumLQRSystem instance
        """
        method_config = config.get("flow_matching", config.get("diffusion", {}))

        name = kwargs.get("name", "pendulum_lqr")
        if dataset_size:
            name = f"pendulum_lqr_{dataset_size}"

        return cls(
            name=name,
            stride=kwargs.get("stride", method_config.get("stride", 1)),
            history_length=kwargs.get(
                "history_length", method_config.get("history_length", 1)
            ),
            horizon_length=kwargs.get(
                "horizon_length", method_config.get("horizon_length", 31)
            ),
            max_path_length=kwargs.get("max_path_length"),
            normalizer=kwargs.get("normalizer"),
            metadata=kwargs.get("metadata"),
        )

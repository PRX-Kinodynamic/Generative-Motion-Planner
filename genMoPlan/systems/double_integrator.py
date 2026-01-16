from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from genMoPlan.datasets.normalization import Normalizer
from genMoPlan.systems.base import BaseSystem, Outcome


class DoubleIntegrator1DSystem(BaseSystem):
    """
    System descriptor for 1D Double Integrator.

    State: [position, velocity]
    - position: x coordinate
    - velocity: x_dot

    This is a simple 2D state space with no angular components.
    The goal is typically to reach the origin (position=0, velocity=0).
    """

    # Class-level defaults for state space
    DEFAULT_STATE_DIM = 2
    DEFAULT_MAX_PATH_LENGTH = 200
    DEFAULT_ANGLE_INDICES: List[int] = []  # No angles in double integrator
    DEFAULT_STATE_NAMES = ["position", "velocity"]

    # Variant-specific limits
    VARIANT_LIMITS = {
        "bang_bang": {
            "mins": [-1.01, -1.01],
            "maxs": [1.01, 1.01],
        },
        "ppo": {
            "mins": [-1.05, -5.0],
            "maxs": [1.05, 5.0],
        },
    }
    DEFAULT_MINS = [-1.01, -1.01]
    DEFAULT_MAXS = [1.01, 1.01]

    def __init__(
        self,
        *,
        name: str = "double_integrator_1d",
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
        success_threshold: float = 0.1,
        variant: str = "bang_bang",
    ):
        # Set defaults from class constants, using variant-specific limits
        if max_path_length is None:
            max_path_length = self.DEFAULT_MAX_PATH_LENGTH
        variant_limits = self.VARIANT_LIMITS.get(variant, {})
        if mins is None:
            mins = variant_limits.get("mins", self.DEFAULT_MINS).copy()
        if maxs is None:
            maxs = variant_limits.get("maxs", self.DEFAULT_MAXS).copy()
        if state_names is None:
            state_names = self.DEFAULT_STATE_NAMES.copy()
        if angle_indices is None:
            angle_indices = self.DEFAULT_ANGLE_INDICES.copy()
        if valid_outcomes is None:
            valid_outcomes = [Outcome.SUCCESS, Outcome.FAILURE, Outcome.INVALID]

        # Set up metadata
        metadata = metadata or {}
        metadata.setdefault("success_threshold", success_threshold)
        metadata.setdefault("variant", variant)

        # No preprocessing needed for double integrator (pure Euclidean)
        trajectory_preprocess_fns = []
        preprocess_kwargs = {
            "trajectory": {
                "angle_indices": [],
            },
            "plan": None,
        }

        # No post-processing needed
        post_process_fns = []
        post_process_fn_kwargs = {}

        super().__init__(
            name=f"{name}_{variant}" if variant else name,
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

    def read_trajectory(self, sequence_path: Union[str, Path]) -> np.ndarray:
        """
        Read a trajectory from a file path.

        Double integrator uses space-separated values instead of comma-separated.

        Args:
            sequence_path: Path to trajectory file

        Returns:
            np.ndarray of shape (T, state_dim) with dtype float32
        """
        # Skip plan files
        if "plan" in str(sequence_path):
            return None

        with open(sequence_path, "r") as f:
            lines = f.readlines()

        trajectory = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line == "":
                if i < len(lines) - 1:
                    raise ValueError(
                        f"[ {self.name} ] Empty line found at {sequence_path} at line {i}"
                    )
                else:
                    break

            # Double integrator uses space-separated values
            state = line.split(" ")
            state = [s for s in state if s != ""]

            if len(state) < self.state_dim:
                raise ValueError(
                    f"[ {self.name} ] Trajectory at {sequence_path} has {len(state)} states at line {i}, expected {self.state_dim}"
                )

            state = state[: self.state_dim]
            state = np.array([float(s) for s in state])
            trajectory.append(state)

        return np.array(trajectory, dtype=np.float32)

    def evaluate_final_state(self, state: np.ndarray) -> Outcome:
        """
        Evaluate the final state of a trajectory.

        Success if the state is close to the origin (position ~= 0, velocity ~= 0).
        """
        threshold = self.metadata.get("success_threshold", 0.1)
        state_norm = np.linalg.norm(state)

        if state_norm <= threshold:
            return Outcome.SUCCESS

        return Outcome.FAILURE

    def evaluate_final_states(self, states: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of final states using vectorized operations.

        Args:
            states: Array of shape (batch_size, state_dim) containing final states

        Returns:
            np.ndarray of shape (batch_size,) with Outcome values
        """
        threshold = self.metadata.get("success_threshold", 0.1)

        # Vectorized norm computation
        state_norms = np.linalg.norm(states, axis=1)

        # Vectorized outcome assignment
        outcomes = np.where(
            state_norms <= threshold,
            Outcome.SUCCESS.value,
            Outcome.FAILURE.value
        )
        return outcomes.astype(np.int32)

    def should_terminate(
        self, state: np.ndarray, t: int, traj_so_far: Optional[np.ndarray]
    ):
        """
        Early termination check.

        Terminate early if the state reaches the origin.
        """
        threshold = self.metadata.get("success_threshold", 0.1)
        state_norm = np.linalg.norm(state)

        if state_norm <= threshold:
            return Outcome.SUCCESS

        return None

    @classmethod
    def create(
        cls,
        stride: int = 1,
        history_length: int = 1,
        horizon_length: int = 31,
        max_path_length: Optional[int] = None,
        variant: str = "bang_bang",
        **kwargs,
    ) -> "DoubleIntegrator1DSystem":
        """
        Factory method to create a DoubleIntegrator1DSystem with sensible defaults.

        Args:
            stride: Stride for trajectory sampling
            history_length: Length of history conditioning
            horizon_length: Length of prediction horizon
            max_path_length: Maximum trajectory length (uses default if None)
            variant: Controller variant ("bang_bang" or "ppo")
            **kwargs: Additional arguments passed to __init__

        Returns:
            DoubleIntegrator1DSystem instance
        """
        return cls(
            stride=stride,
            history_length=history_length,
            horizon_length=horizon_length,
            max_path_length=max_path_length,
            variant=variant,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        variant: str = "bang_bang",
        **kwargs,
    ) -> "DoubleIntegrator1DSystem":
        """
        Create a DoubleIntegrator1DSystem from a config dictionary.

        Args:
            config: Configuration dictionary
            variant: Controller variant ("bang_bang" or "ppo")
            **kwargs: Additional arguments to override config values

        Returns:
            DoubleIntegrator1DSystem instance
        """
        method_config = config.get("flow_matching", config.get("diffusion", {}))

        return cls(
            stride=kwargs.get("stride", method_config.get("stride", 1)),
            history_length=kwargs.get(
                "history_length", method_config.get("history_length", 1)
            ),
            horizon_length=kwargs.get(
                "horizon_length", method_config.get("horizon_length", 31)
            ),
            max_path_length=kwargs.get("max_path_length"),
            variant=variant,
            normalizer=kwargs.get("normalizer"),
            metadata=kwargs.get("metadata"),
        )

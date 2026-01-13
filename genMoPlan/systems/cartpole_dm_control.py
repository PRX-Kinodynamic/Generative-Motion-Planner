from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from genMoPlan.datasets.normalization import Normalizer
from genMoPlan.systems.base import BaseSystem, Outcome
from genMoPlan.utils.data_processing import (
    handle_angle_wraparound,
    augment_unwrapped_state_data,
    shift_to_zero_center_angles,
)
from genMoPlan.utils.trajectory import process_angles


def _create_cartpole_dm_manifold():
    """Create the manifold for cartpole DM control (lazy import to avoid circular deps)."""
    from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product

    return Product(
        input_dim=4,
        manifolds=[(Euclidean(), 1), (FlatTorus(), 1), (Euclidean(), 2)],
    )


class CartpoleDMControlSystem(BaseSystem):
    """
    System descriptor for Cartpole DM Control environment.

    State: [x, theta, x_dot, theta_dot]
    - x: cart position
    - theta: pole angle (wrapped to [-pi, pi])
    - x_dot: cart velocity
    - theta_dot: pole angular velocity

    Success is defined as reaching a state close to the origin (upright position).
    """

    # Class-level defaults for state space
    DEFAULT_STATE_DIM = 4
    DEFAULT_MINS = [-2.0699819, -np.pi, -8.62452465, -18.5882527]
    DEFAULT_MAXS = [2.0699819, np.pi, 8.62452465, 18.5882527]
    DEFAULT_MAX_PATH_LENGTH = 1001
    DEFAULT_ANGLE_INDICES = [1]
    DEFAULT_STATE_NAMES = ["x", "theta", "x_dot", "theta_dot"]

    def __init__(
        self,
        *,
        name: str = "cartpole_dm_control",
        state_dim: int = 4,
        stride: int,
        history_length: int,
        horizon_length: int,
        max_path_length: Optional[int] = None,
        mins: Optional[List[float]] = None,
        maxs: Optional[List[float]] = None,
        state_names: Optional[List[str]] = None,
        angle_indices: Optional[List[int]] = None,
        manifold: Optional[Any] = None,
        valid_outcomes: Optional[Sequence[Outcome]] = None,
        normalizer: Optional[Normalizer] = None,
        metadata: Optional[Dict[str, Any]] = None,
        success_threshold: float = 1.0,
        use_manifold: bool = False,
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
            valid_outcomes = [Outcome.SUCCESS, Outcome.FAILURE, Outcome.INVALID]

        # Create manifold if using manifold-based flow matching
        if use_manifold and manifold is None:
            manifold = _create_cartpole_dm_manifold()

        # Set up metadata
        metadata = metadata or {}
        metadata.setdefault("success_threshold", success_threshold)
        metadata.setdefault("angle_indices", angle_indices)

        # Preprocessing functions for diffusion (Euclidean)
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

        # Post-processing for inference
        post_process_fns = [process_angles]
        post_process_fn_kwargs = {"angle_indices": angle_indices}

        # Manifold unwrap functions
        manifold_unwrap_fns = [shift_to_zero_center_angles]
        manifold_unwrap_kwargs = {"angle_indices": angle_indices}

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
            manifold=manifold,
            trajectory_preprocess_fns=trajectory_preprocess_fns,
            preprocess_kwargs=preprocess_kwargs,
            post_process_fns=post_process_fns,
            post_process_fn_kwargs=post_process_fn_kwargs,
            manifold_unwrap_fns=manifold_unwrap_fns,
            manifold_unwrap_kwargs=manifold_unwrap_kwargs,
            valid_outcomes=valid_outcomes,
            normalizer=normalizer,
            metadata=metadata,
        )

    def evaluate_final_state(self, state: np.ndarray) -> Outcome:
        """
        Evaluate the final state of a trajectory.

        Success if the state norm is below the threshold (close to upright).
        """
        threshold = self.metadata.get("success_threshold", 1.0)
        state_norm = np.linalg.norm(state)

        if state_norm <= threshold:
            return Outcome.SUCCESS

        # Out-of-bounds is treated as a failure outcome here; INVALID is reserved
        # for uncertain classifications at the RoA layer.
        return Outcome.FAILURE

    def should_terminate(
        self, state: np.ndarray, t: int, traj_so_far: Optional[np.ndarray]
    ):
        """
        Early termination check.

        Terminate early if the cart goes out of bounds.
        """
        x_limit = self.DEFAULT_MAXS[0]

        if abs(state[0]) > x_limit:
            return Outcome.FAILURE

        return None

    @classmethod
    def create(
        cls,
        stride: int = 1,
        history_length: int = 1,
        horizon_length: int = 31,
        max_path_length: Optional[int] = None,
        use_manifold: bool = False,
        **kwargs,
    ) -> "CartpoleDMControlSystem":
        """
        Factory method to create a CartpoleDMControlSystem with sensible defaults.

        Args:
            stride: Stride for trajectory sampling
            history_length: Length of history conditioning
            horizon_length: Length of prediction horizon
            max_path_length: Maximum trajectory length (uses default if None)
            use_manifold: Whether to use manifold-based flow matching
            **kwargs: Additional arguments passed to __init__

        Returns:
            CartpoleDMControlSystem instance
        """
        return cls(
            stride=stride,
            history_length=history_length,
            horizon_length=horizon_length,
            max_path_length=max_path_length,
            use_manifold=use_manifold,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        **kwargs,
    ) -> "CartpoleDMControlSystem":
        """
        Create a CartpoleDMControlSystem from a config dictionary.

        This method extracts training parameters (stride, history_length, etc.)
        from the config while using system defaults for state space properties.

        Args:
            config: Configuration dictionary (typically from config files)
            **kwargs: Additional arguments to override config values

        Returns:
            CartpoleDMControlSystem instance
        """
        method_config = config.get("flow_matching", config.get("diffusion", {}))
        inference_config = config.get("inference", {})
        use_manifold = "manifold" in method_config and method_config.get("manifold") is not None

        return cls(
            name=kwargs.get("name", "cartpole_dm_control"),
            stride=kwargs.get("stride", method_config.get("stride", 1)),
            history_length=kwargs.get(
                "history_length", method_config.get("history_length", 1)
            ),
            horizon_length=kwargs.get(
                "horizon_length", method_config.get("horizon_length", 31)
            ),
            max_path_length=kwargs.get("max_path_length"),
            success_threshold=kwargs.get(
                "success_threshold",
                inference_config.get("success_threshold", 1.0),
            ),
            use_manifold=kwargs.get("use_manifold", use_manifold),
            normalizer=kwargs.get("normalizer"),
            metadata=kwargs.get("metadata"),
        )

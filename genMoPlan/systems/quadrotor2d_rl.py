from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from genMoPlan.datasets.normalization import Normalizer
from genMoPlan.systems.base import BaseSystem, Outcome
from genMoPlan.utils.data_processing import (
    handle_angle_wraparound,
    augment_unwrapped_state_data,
)
from genMoPlan.utils.trajectory import process_angles


def _create_quadrotor2d_manifold():
    """Create the manifold for 2D quadrotor (lazy import to avoid circular deps).

    State: [x, z, theta, x_dot, z_dot, theta_dot]
    Manifold: R^2 (position) x S^1 (theta) x R^3 (velocities)
    """
    from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product

    return Product(
        input_dim=6,
        manifolds=[(Euclidean(), 2), (FlatTorus(), 1), (Euclidean(), 3)],
    )


class Quadrotor2DRLSystem(BaseSystem):
    """
    System descriptor for 2D Quadrotor RL environment (PyBullet).

    State: [x, z, theta, x_dot, z_dot, theta_dot]
    - x: horizontal position (m)
    - z: altitude/vertical position (m)
    - theta: pitch angle (rad, wrapped to [-pi, pi])
    - x_dot: horizontal velocity (m/s)
    - z_dot: vertical velocity (m/s)
    - theta_dot: pitch rate (rad/s)

    Goal: Stabilize to hover at (x=0, z=1, theta=0, velocities=0).
    Success: Final state within Euclidean distance 0.2 of goal.
    """

    # Class-level defaults for state space
    DEFAULT_STATE_DIM = 6
    DEFAULT_MINS = [-1.0, 0.1, -np.pi, -1.0, -1.0, -8.0]
    DEFAULT_MAXS = [1.0, 1.5, np.pi, 1.0, 1.0, 8.0]
    DEFAULT_MAX_PATH_LENGTH = 709  # From dataset statistics
    DEFAULT_ANGLE_INDICES = [2]  # theta is the angle
    DEFAULT_STATE_NAMES = ["x", "z", "theta", "x_dot", "z_dot", "theta_dot"]

    # Goal state for stabilization
    GOAL_STATE = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Success threshold (Euclidean distance from goal)
    SUCCESS_RADIUS = 0.2

    # Failure limits (from dataset termination thresholds)
    # Note: theta has no termination bound (inf), so we don't check it
    FAILURE_LIMITS = np.array([1.0, np.inf, np.inf, 1.0, 1.0, 8.0], dtype=np.float32)
    # Z has asymmetric bounds: min=0.1, max=1.5
    Z_MIN = 0.1
    Z_MAX = 1.5

    def __init__(
        self,
        *,
        name: str = "quadrotor2d_rl",
        dataset: str,  # REQUIRED - for loading achieved bounds
        state_dim: int = 6,
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
        success_radius: float = 0.2,
        use_manifold: bool = False,
    ):
        # Set defaults from class constants
        if max_path_length is None:
            max_path_length = self.DEFAULT_MAX_PATH_LENGTH
        if state_names is None:
            state_names = self.DEFAULT_STATE_NAMES.copy()
        if angle_indices is None:
            angle_indices = self.DEFAULT_ANGLE_INDICES.copy()
        if valid_outcomes is None:
            valid_outcomes = [Outcome.SUCCESS, Outcome.FAILURE, Outcome.INVALID]

        # Always create the TRUE manifold - reflects actual state space topology
        if manifold is None:
            from genMoPlan.utils.manifold import ManifoldWrapper

            raw_manifold = _create_quadrotor2d_manifold()
            manifold = ManifoldWrapper(raw_manifold)

        # model_manifold: what the generative model uses for its architecture
        model_manifold = manifold if use_manifold else None

        # Set up metadata
        metadata = metadata or {}
        metadata.setdefault("success_radius", success_radius)
        metadata.setdefault("goal_state", self.GOAL_STATE.tolist())
        metadata.setdefault("angle_indices", angle_indices)
        metadata.setdefault("failure_limits", self.FAILURE_LIMITS.tolist())
        metadata.setdefault("z_min", self.Z_MIN)
        metadata.setdefault("z_max", self.Z_MAX)
        metadata.setdefault("invalid_label", -1)
        metadata.setdefault("invalid_labels", [metadata["invalid_label"]])
        metadata.setdefault("invalid_outcomes", ["INVALID"])

        # Preprocessing functions depend on whether using manifold
        if use_manifold:
            trajectory_preprocess_fns = []
            preprocess_kwargs = {
                "trajectory": {},
                "plan": None,
            }
        else:
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

        super().__init__(
            name=name,
            dataset=dataset,
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
            model_manifold=model_manifold,
            trajectory_preprocess_fns=trajectory_preprocess_fns,
            preprocess_kwargs=preprocess_kwargs,
            post_process_fns=post_process_fns,
            post_process_fn_kwargs=post_process_fn_kwargs,
            valid_outcomes=valid_outcomes,
            normalizer=normalizer,
            metadata=metadata,
        )

    def evaluate_final_state(self, state: np.ndarray) -> Outcome:
        """
        Evaluate the final state of a trajectory.

        Delegates to evaluate_final_states for consistency.
        """
        state_vec = self._as_state_vector(state)
        outcomes = self.evaluate_final_states(state_vec[np.newaxis, :])
        return Outcome(outcomes[0])

    def evaluate_final_states(self, states: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of final states using vectorized, manifold-aware operations.

        Success: Within Euclidean distance SUCCESS_RADIUS from GOAL_STATE.
        Failure: Out of bounds (position/velocity limits exceeded).

        Args:
            states: Array of shape (batch_size, state_dim) containing final states

        Returns:
            np.ndarray of shape (batch_size,) with Outcome values
        """
        # Handle batched states - ensure 2D
        if states.ndim == 1:
            states = states[np.newaxis, :]

        success_radius = self.metadata.get("success_radius", self.SUCCESS_RADIUS)
        failure_limits = np.asarray(
            self.metadata.get("failure_limits", self.FAILURE_LIMITS), dtype=np.float32
        )
        z_min = self.metadata.get("z_min", self.Z_MIN)
        z_max = self.metadata.get("z_max", self.Z_MAX)
        goal_state = np.asarray(
            self.metadata.get("goal_state", self.GOAL_STATE), dtype=np.float32
        )

        # Use manifold-aware distance to goal
        states_t = torch.from_numpy(states).float()
        goal_t = torch.from_numpy(goal_state).float().expand_as(states_t)
        per_dim_dist = self.manifold.dist(states_t, goal_t)  # Geodesic for theta

        # Check for success: Euclidean distance from goal
        # For position/velocity dimensions, use standard Euclidean distance
        # For theta (index 2), the manifold.dist already gives geodesic distance
        euclidean_dims = per_dim_dist[:, [0, 1, 3, 4, 5]]  # x, z, x_dot, z_dot, theta_dot
        angular_dim = per_dim_dist[:, 2]  # theta

        total_dist = torch.sqrt(
            torch.sum(euclidean_dims**2, dim=1) + angular_dim**2
        )
        is_within_success = total_dist < success_radius

        # Check for out of bounds
        # Distance from origin for failure limit check
        origin_t = torch.zeros_like(states_t)
        per_dim_dist_origin = self.manifold.dist(states_t, origin_t)

        failure_limits_t = torch.from_numpy(failure_limits).float()
        is_over_limit = torch.any(per_dim_dist_origin > failure_limits_t, dim=1)

        # Special handling for z with asymmetric bounds
        z_values = states_t[:, 1]
        is_z_out = (z_values < z_min) | (z_values > z_max)

        is_out_of_bounds = is_over_limit | is_z_out

        # Assign outcomes: FAILURE > SUCCESS > INVALID (priority order)
        outcomes = torch.full(
            (states.shape[0],), Outcome.INVALID.value, dtype=torch.int32
        )
        outcomes[is_within_success] = Outcome.SUCCESS.value
        outcomes[is_out_of_bounds] = Outcome.FAILURE.value  # Overrides SUCCESS if both

        return outcomes.detach().cpu().numpy()

    def _as_state_vector(self, state: np.ndarray) -> np.ndarray:
        state_np = np.asarray(state, dtype=np.float32)
        return state_np[0] if state_np.ndim > 1 else state_np

    @classmethod
    def create(
        cls,
        dataset: str,  # REQUIRED
        stride: int = 1,
        history_length: int = 1,
        horizon_length: int = 31,
        max_path_length: Optional[int] = None,
        use_manifold: bool = False,
        **kwargs,
    ) -> "Quadrotor2DRLSystem":
        """
        Factory method to create a Quadrotor2DRLSystem with sensible defaults.

        Args:
            dataset: Name of the dataset (required for loading achieved bounds)
            stride: Stride for trajectory sampling
            history_length: Length of history conditioning
            horizon_length: Length of prediction horizon
            max_path_length: Maximum trajectory length (uses default if None)
            use_manifold: Whether to use manifold-based flow matching
            **kwargs: Additional arguments passed to __init__

        Returns:
            Quadrotor2DRLSystem instance
        """
        return cls(
            dataset=dataset,
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
        dataset: str,  # REQUIRED
        **kwargs,
    ) -> "Quadrotor2DRLSystem":
        """
        Create a Quadrotor2DRLSystem from a config dictionary.

        Args:
            config: Configuration dictionary (typically from config files)
            dataset: Name of the dataset (required for loading achieved bounds)
            **kwargs: Additional arguments to override config values

        Returns:
            Quadrotor2DRLSystem instance
        """
        method_config = config.get("flow_matching", config.get("diffusion", {}))
        use_manifold = (
            "manifold" in method_config and method_config.get("manifold") is not None
        )

        return cls(
            name=kwargs.get("name", "quadrotor2d_rl"),
            dataset=dataset,
            stride=kwargs.get("stride", method_config.get("stride", 1)),
            history_length=kwargs.get(
                "history_length", method_config.get("history_length", 1)
            ),
            horizon_length=kwargs.get(
                "horizon_length", method_config.get("horizon_length", 31)
            ),
            max_path_length=kwargs.get("max_path_length"),
            use_manifold=kwargs.get("use_manifold", use_manifold),
            normalizer=kwargs.get("normalizer"),
            metadata=kwargs.get("metadata"),
        )

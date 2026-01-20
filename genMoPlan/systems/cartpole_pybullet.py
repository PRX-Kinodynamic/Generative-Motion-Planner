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


def _create_cartpole_manifold():
    """Create the manifold for cartpole (lazy import to avoid circular deps)."""
    from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product

    return Product(
        input_dim=4,
        manifolds=[(Euclidean(), 1), (FlatTorus(), 1), (Euclidean(), 2)],
    )


class CartpolePyBulletSystem(BaseSystem):
    """
    System descriptor for Cartpole PyBullet environment.

    State: [x, theta, x_dot, theta_dot]
    - x: cart position
    - theta: pole angle (wrapped to [-pi, pi])
    - x_dot: cart velocity
    - theta_dot: pole angular velocity

    Success requires keeping all state variables within a tight tolerance
    around the origin.
    """

    # Class-level defaults for state space
    DEFAULT_STATE_DIM = 4
    DEFAULT_MINS = [-6.0, -np.pi, -5.0, -5.0]
    DEFAULT_MAXS = [6.0, np.pi, 5.0, 5.0]
    DEFAULT_MAX_PATH_LENGTH = 613
    DEFAULT_ANGLE_INDICES = [1]
    DEFAULT_STATE_NAMES = ["x", "theta", "x_dot", "theta_dot"]

    # Success/failure thresholds
    SUCCESS_TOLERANCES = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    FAILURE_LIMITS = np.array([5.9, np.inf, 4.9, 4.9], dtype=np.float32)

    def __init__(
        self,
        *,
        name: str = "cartpole_pybullet",
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

        # Manifold unwrap functions
        manifold_unwrap_fns = [shift_to_zero_center_angles]
        manifold_unwrap_kwargs = {"angle_indices": angle_indices}

        # Create manifold if using manifold-based flow matching
        if use_manifold and manifold is None:
            from genMoPlan.utils.manifold import ManifoldWrapper
            raw_manifold = _create_cartpole_manifold()
            manifold = ManifoldWrapper(
                raw_manifold,
                manifold_unwrap_fns=manifold_unwrap_fns,
                manifold_unwrap_kwargs=manifold_unwrap_kwargs
            )

        # Set up metadata
        metadata = metadata or {}
        metadata.setdefault("success_threshold", success_threshold)
        metadata.setdefault("angle_indices", angle_indices)
        metadata.setdefault("success_tolerances", self.SUCCESS_TOLERANCES.tolist())
        metadata.setdefault("failure_limits", self.FAILURE_LIMITS.tolist())
        metadata.setdefault("invalid_label", -1)
        metadata.setdefault("invalid_labels", [metadata["invalid_label"]])
        metadata.setdefault("invalid_outcomes", ["INVALID"])

        # Preprocessing functions depend on whether using manifold
        # - Manifold flow matching: manifold handles angle topology natively
        # - Euclidean (diffusion): needs unwrapping and augmentation
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
            # Unwrap functions are already in ManifoldWrapper if manifold is set
            manifold_unwrap_fns=manifold_unwrap_fns if manifold is None else None,
            manifold_unwrap_kwargs=manifold_unwrap_kwargs if manifold is None else None,
            valid_outcomes=valid_outcomes,
            normalizer=normalizer,
            metadata=metadata,
        )

    def evaluate_final_state(self, state: np.ndarray) -> Outcome:
        """
        Evaluate the final state of a trajectory.

        Success if the final state is within the tolerance bounds.
        Failure if any variable exceeds its bounds.
        Otherwise, the state is marked invalid.
        """
        state_vec = self._as_state_vector(state)

        if self._is_out_of_bounds(state_vec):
            # Out-of-bounds states are treated as failures.
            return Outcome.FAILURE

        if self._is_within_success_tolerance(state_vec):
            return Outcome.SUCCESS

        return Outcome.INVALID

    def evaluate_final_states(self, states: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of final states using vectorized operations.

        Args:
            states: Array of shape (batch_size, state_dim) containing final states

        Returns:
            np.ndarray of shape (batch_size,) with Outcome values
        """
        # Handle batched states - ensure 2D
        if states.ndim == 1:
            states = states[np.newaxis, :]

        failure_limits = np.asarray(
            self.metadata.get("failure_limits", self.FAILURE_LIMITS), dtype=np.float32
        )
        success_tolerances = np.asarray(
            self.metadata.get("success_tolerances", self.SUCCESS_TOLERANCES), dtype=np.float32
        )

        # Vectorized bounds check: any state element exceeds failure limits
        abs_states = np.abs(states)
        is_out_of_bounds = np.any(abs_states > failure_limits, axis=1)

        # Vectorized success check: all state elements within success tolerances
        is_within_success = np.all(abs_states < success_tolerances, axis=1)

        # Assign outcomes: FAILURE > SUCCESS > INVALID (priority order)
        outcomes = np.full(states.shape[0], Outcome.INVALID.value, dtype=np.int32)
        outcomes[is_within_success] = Outcome.SUCCESS.value
        outcomes[is_out_of_bounds] = Outcome.FAILURE.value  # Overrides SUCCESS if both true

        return outcomes

    def _as_state_vector(self, state: np.ndarray) -> np.ndarray:
        state_np = np.asarray(state, dtype=np.float32)
        return state_np[0] if state_np.ndim > 1 else state_np

    def _is_out_of_bounds(self, state: np.ndarray) -> bool:
        limits = np.asarray(
            self.metadata.get("failure_limits", self.FAILURE_LIMITS), dtype=np.float32
        )
        return bool(np.any(np.abs(state) > limits))

    def _is_within_success_tolerance(self, state: np.ndarray) -> bool:
        tolerances = np.asarray(
            self.metadata.get("success_tolerances", self.SUCCESS_TOLERANCES),
            dtype=np.float32,
        )
        return bool(np.all(np.abs(state) < tolerances))

    @classmethod
    def create(
        cls,
        stride: int = 1,
        history_length: int = 1,
        horizon_length: int = 31,
        max_path_length: Optional[int] = None,
        use_manifold: bool = False,
        **kwargs,
    ) -> "CartpolePyBulletSystem":
        """
        Factory method to create a CartpolePyBulletSystem with sensible defaults.

        Args:
            stride: Stride for trajectory sampling
            history_length: Length of history conditioning
            horizon_length: Length of prediction horizon
            max_path_length: Maximum trajectory length (uses default if None)
            use_manifold: Whether to use manifold-based flow matching
            **kwargs: Additional arguments passed to __init__

        Returns:
            CartpolePyBulletSystem instance
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
    ) -> "CartpolePyBulletSystem":
        """
        Create a CartpolePyBulletSystem from a config dictionary.

        This method extracts training parameters (stride, history_length, etc.)
        from the config while using system defaults for state space properties.

        Args:
            config: Configuration dictionary (typically from config files)
            **kwargs: Additional arguments to override config values

        Returns:
            CartpolePyBulletSystem instance
        """
        method_config = config.get("flow_matching", config.get("diffusion", {}))
        use_manifold = "manifold" in method_config and method_config.get("manifold") is not None

        return cls(
            name=kwargs.get("name", "cartpole_pybullet"),
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

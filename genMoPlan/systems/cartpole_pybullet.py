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
    FAILURE_LIMITS = np.array([5.5, np.inf, 4.5, 4.5], dtype=np.float32)

    ATTRACTOR_RADIUS = 0.2

    def __init__(
        self,
        *,
        name: str = "cartpole_pybullet",
        dataset: str,  # REQUIRED - for loading achieved bounds
        state_dim: int = 4,
        stride: int,
        history_length: int,
        horizon_length: int,
        max_path_length: Optional[int] = None,
        mins: Optional[List[float]] = None,  # Optional override (normally from achieved bounds)
        maxs: Optional[List[float]] = None,  # Optional override (normally from achieved bounds)
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
        # mins/maxs: don't set defaults here - let parent load from achieved bounds
        if state_names is None:
            state_names = self.DEFAULT_STATE_NAMES.copy()
        if angle_indices is None:
            angle_indices = self.DEFAULT_ANGLE_INDICES.copy()
        if valid_outcomes is None:
            valid_outcomes = [Outcome.SUCCESS, Outcome.FAILURE, Outcome.INVALID]

        # Always create the TRUE manifold - reflects actual state space topology
        # Used for distance computations, projection, and evaluation metrics
        if manifold is None:
            from genMoPlan.utils.manifold import ManifoldWrapper

            raw_manifold = _create_cartpole_manifold()
            manifold = ManifoldWrapper(raw_manifold)

        # model_manifold: what the generative model uses for its architecture
        # - When use_manifold=True: model operates on manifold (GeodesicProbPath, RiemannianODESolver)
        # - When use_manifold=False: model operates in Euclidean space (model_manifold=None)
        model_manifold = manifold if use_manifold else None

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

        Uses manifold.dist() for geodesic distance computation on angular components.
        This correctly handles angle wrapping for the theta dimension.

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
            self.metadata.get("success_tolerances", self.SUCCESS_TOLERANCES),
            dtype=np.float32,
        )

        # Use manifold-aware distance to origin
        # Convert to torch for manifold.dist() (library is torch-based)
        states_t = torch.from_numpy(states).float()
        origin_t = torch.zeros_like(states_t)
        per_dim_dist = self.manifold.dist(states_t, origin_t)  # Geodesic for angles

        # Check for out of bounds
        failure_limits_t = torch.from_numpy(failure_limits).float()
        is_out_of_bounds = torch.any(per_dim_dist > failure_limits_t, dim=1)

        # Check for success

        # success_tolerances_t = torch.from_numpy(success_tolerances).float()
        # # Vectorized success check: all state elements within success tolerances
        # # is_within_success = torch.all(per_dim_dist < success_tolerances_t, dim=1)

        euclidean_dist = per_dim_dist[:, [0, 2, 3]]
        angular_dist = per_dim_dist[:, 1]

        dist = torch.sqrt(torch.sum(euclidean_dist**2, dim=1) + angular_dist**2)

        is_within_success = dist < self.ATTRACTOR_RADIUS

        # Assign outcomes: FAILURE > SUCCESS > INVALID (priority order)
        outcomes = torch.full(
            (states.shape[0],), Outcome.INVALID.value, dtype=torch.int32
        )
        outcomes[is_within_success] = Outcome.SUCCESS.value
        outcomes[is_out_of_bounds] = (
            Outcome.FAILURE.value
        )  # Overrides SUCCESS if both true

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
    ) -> "CartpolePyBulletSystem":
        """
        Factory method to create a CartpolePyBulletSystem with sensible defaults.

        Args:
            dataset: Name of the dataset (required for loading achieved bounds)
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
    ) -> "CartpolePyBulletSystem":
        """
        Create a CartpolePyBulletSystem from a config dictionary.

        This method extracts training parameters (stride, history_length, etc.)
        from the config while using system defaults for state space properties.

        Args:
            config: Configuration dictionary (typically from config files)
            dataset: Name of the dataset (required for loading achieved bounds)
            **kwargs: Additional arguments to override config values

        Returns:
            CartpolePyBulletSystem instance
        """
        method_config = config.get("flow_matching", config.get("diffusion", {}))
        use_manifold = (
            "manifold" in method_config and method_config.get("manifold") is not None
        )

        return cls(
            name=kwargs.get("name", "cartpole_pybullet"),
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

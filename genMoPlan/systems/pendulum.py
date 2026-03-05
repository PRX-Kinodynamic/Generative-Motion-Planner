from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from genMoPlan.datasets.normalization import Normalizer
from genMoPlan.systems.base import BaseSystem, Outcome
from genMoPlan.utils.data_processing import (
    handle_angle_wraparound,
    augment_unwrapped_state_data,
)


def _create_pendulum_manifold():
    """Create the manifold for pendulum (lazy import to avoid circular deps)."""
    from flow_matching.utils.manifolds import FlatTorus, Euclidean, Product

    return Product(
        input_dim=2,
        manifolds=[(FlatTorus(), 1), (Euclidean(), 1)],
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
    SUCCESS_THRESHOLD = 0.075

    def __init__(
        self,
        *,
        name: str = "pendulum_lqr",
        dataset: str,  # REQUIRED - for loading achieved bounds
        state_dim: int = 2,
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
        success_threshold: float = 0.1,
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
            # Pendulum trajectories are ultimately classified as success or failure;
            # INVALID can be used by higher-level analysis when labels are uncertain.
            valid_outcomes = [Outcome.SUCCESS, Outcome.FAILURE, Outcome.INVALID]

        # Always create the TRUE manifold - reflects actual state space topology
        # Used for distance computations, projection, and evaluation metrics
        if manifold is None:
            from genMoPlan.utils.manifold import ManifoldWrapper
            raw_manifold = _create_pendulum_manifold()
            manifold = ManifoldWrapper(raw_manifold)

        # model_manifold: what the generative model uses for its architecture
        # - When use_manifold=True: model operates on manifold (GeodesicProbPath, RiemannianODESolver)
        # - When use_manifold=False: model operates in Euclidean space (model_manifold=None)
        model_manifold = manifold if use_manifold else None

        # Set up metadata
        metadata = metadata or {}
        metadata.setdefault("success_threshold", success_threshold)
        metadata.setdefault("angle_indices", angle_indices)

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

        # No special post-processing for pendulum (angles stay in natural range)
        post_process_fns = []
        post_process_fn_kwargs = {}

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
        states = np.asarray(state, dtype=np.float32)
        if states.ndim == 1:
            states = states[np.newaxis, :]
        outcomes = self.evaluate_final_states(states)
        return Outcome(outcomes[0])

    def evaluate_final_states(self, states: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of final states using vectorized, manifold-aware operations.

        Uses manifold.dist() for geodesic distance computation on the theta angle.
        This correctly handles angle wrapping regardless of convention.

        Args:
            states: Array of shape (batch_size, state_dim) containing final states

        Returns:
            np.ndarray of shape (batch_size,) with Outcome values
        """
        threshold = self.metadata.get("success_threshold", self.SUCCESS_THRESHOLD)

        # Use manifold-aware distance to origin (upright position)
        # Convert to torch for manifold.dist() (library is torch-based)
        states_t = torch.from_numpy(states).float()
        origin_t = torch.zeros_like(states_t)
        per_dim_dist = self.manifold.dist(states_t, origin_t)  # Geodesic for angle

        # Compute state norm from manifold distances
        upright_distance = torch.sqrt(torch.sum(per_dim_dist**2, dim=1))

        # Vectorized outcome assignment
        outcomes = torch.where(
            upright_distance <= threshold,
            Outcome.SUCCESS.value,
            Outcome.FAILURE.value
        )
        return outcomes.numpy().astype(np.int32)

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
    ) -> "PendulumLQRSystem":
        """
        Factory method to create a PendulumLQRSystem with sensible defaults.

        Args:
            dataset: Name of the dataset (required for loading achieved bounds)
            stride: Stride for trajectory sampling
            history_length: Length of history conditioning
            horizon_length: Length of prediction horizon
            max_path_length: Maximum trajectory length (uses default if None)
            use_manifold: Whether to use manifold-based flow matching
            **kwargs: Additional arguments passed to __init__

        Returns:
            PendulumLQRSystem instance
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
        dataset_size: Optional[str] = None,
        use_manifold: bool = False,
        **kwargs,
    ) -> "PendulumLQRSystem":
        """
        Create a PendulumLQRSystem from a config dictionary.

        This method extracts training parameters (stride, history_length, etc.)
        from the config while using system defaults for state space properties.

        Args:
            config: Configuration dictionary
            dataset: Name of the dataset (required for loading achieved bounds)
            dataset_size: Optional dataset size indicator (e.g., "5k" or "50k")
            use_manifold: Whether to use manifold-based flow matching
            **kwargs: Additional arguments to override config values

        Returns:
            PendulumLQRSystem instance
        """
        method_config = config.get("flow_matching", config.get("diffusion", {}))

        # Detect use_manifold from config if not explicitly provided
        if not use_manifold:
            use_manifold = method_config.get("use_manifold", False) or (
                "manifold" in method_config and method_config.get("manifold") is not None
            )

        name = kwargs.get("name", "pendulum_lqr")
        if dataset_size:
            name = f"pendulum_lqr_{dataset_size}"

        return cls(
            name=name,
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

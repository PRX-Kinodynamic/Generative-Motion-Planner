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


def _create_humanoid_manifold():
    """Create the manifold for humanoid (lazy import to avoid circular deps)."""
    from flow_matching.utils.manifolds import Euclidean, Product, Sphere

    return Product(
        input_dim=67,
        manifolds=[(Euclidean(), 34), (Sphere(), 3), (Euclidean(), 30)],
    )


class HumanoidGetUpSystem(BaseSystem):
    """
    System descriptor for Humanoid Get-Up task.

    State dimension: 67
    - Positions (34 dims): body positions and joint angles
    - Orientation (3 dims): represented on sphere manifold
    - Velocities (30 dims): body velocities and joint velocities

    The task is for the humanoid to get up from a lying position.
    """

    # Class-level defaults for state space
    DEFAULT_STATE_DIM = 67
    DEFAULT_MAX_PATH_LENGTH = 745
    DEFAULT_ANGLE_INDICES: List[int] = []  # Humanoid uses sphere manifold, not angles
    DEFAULT_STATE_NAMES: List[str] = []  # Too many to list

    # State limits (from humanoid dataset)
    DEFAULT_MINS = [
        -0.859485, -1.375191, -0.688990, -0.525767, -0.632861, -1.994933, -2.905200,
        -0.540604, -1.030145, -0.544243, -0.614817, -1.993659, -2.883713, -0.487868,
        -1.017001, -1.579886, -1.602320, -1.720383, -1.183831, -1.170331, -1.739313,
        0.051704, -0.455080, -0.168303, -0.557510, -0.858317, -0.986132, -1.281937,
        -0.446948, -0.786912, -0.556428, -0.895774, -1.038971, -1.279955, -1.000000,
        -0.999984, -0.896724, -1.914248, -1.915843, -3.327530, -2.658107, -2.528388,
        -5.331130, -11.301671, -13.006869, -17.815111, -14.461623, -14.078833,
        -12.893085, -12.061430, -12.384354, -12.161468, -17.793238, -26.302641,
        -20.862368, -10.522175, -13.214445, -13.203420, -20.225025, -19.451426,
        -21.217875, -24.788021, -18.834494, -21.369850, -22.102814, -20.644373,
        -21.245199,
    ]
    DEFAULT_MAXS = [
        0.838374, 0.609621, 0.725149, 0.164440, 0.517798, 0.522608, 0.185058,
        0.921429, 0.983391, 0.163125, 0.545136, 0.491036, 0.209521, 0.921906,
        1.020330, 1.199831, 1.143854, 0.985109, 1.646968, 1.620824, 0.964722,
        1.494286, 0.617372, 0.787087, 0.556917, 1.055067, 1.020654, 0.523082,
        0.618061, 0.187432, 0.555102, 1.036542, 0.980308, 0.523863, 1.000000,
        1.000000, 1.000000, 1.900168, 1.837195, 1.918130, 2.789604, 2.940170,
        2.997349, 8.091124, 12.754802, 14.861890, 11.852520, 14.249229, 11.141454,
        11.107437, 12.403696, 14.443352, 25.315464, 19.668034, 18.593166, 11.033415,
        12.674791, 17.079679, 24.056908, 22.487865, 21.304333, 23.978119, 19.407972,
        24.520531, 25.825670, 21.328411, 24.432425,
    ]

    # Sphere manifold indices (last 3 dims of position section)
    SPHERE_INDICES = [34, 35, 36]

    def __init__(
        self,
        *,
        name: str = "humanoid_get_up",
        state_dim: int = 67,
        stride: int,
        history_length: int,
        horizon_length: int,
        max_path_length: Optional[int] = None,
        mins: Optional[List[float]] = None,
        maxs: Optional[List[float]] = None,
        angle_indices: Optional[List[int]] = None,
        manifold: Optional[Any] = None,
        valid_outcomes: Optional[Sequence[Outcome]] = None,
        normalizer: Optional[Normalizer] = None,
        metadata: Optional[Dict[str, Any]] = None,
        height_threshold: float = 0.8,
        use_manifold: bool = False,
    ):
        # Set defaults from class constants
        if max_path_length is None:
            max_path_length = self.DEFAULT_MAX_PATH_LENGTH
        if mins is None:
            mins = self.DEFAULT_MINS.copy()
        if maxs is None:
            maxs = self.DEFAULT_MAXS.copy()
        if angle_indices is None:
            angle_indices = self.DEFAULT_ANGLE_INDICES.copy()
        if valid_outcomes is None:
            # Humanoid trajectories are ultimately classified as success or failure;
            # INVALID can be used by higher-level analysis when labels are uncertain.
            valid_outcomes = [Outcome.SUCCESS, Outcome.FAILURE, Outcome.INVALID]

        # Manifold unwrap functions (use index 0 for sphere coords)
        manifold_unwrap_fns = [shift_to_zero_center_angles]
        manifold_unwrap_kwargs = {"angle_indices": [0]}

        # Create manifold if using manifold-based flow matching
        if use_manifold and manifold is None:
            from genMoPlan.utils.manifold import ManifoldWrapper
            raw_manifold = _create_humanoid_manifold()
            manifold = ManifoldWrapper(
                raw_manifold,
                manifold_unwrap_fns=manifold_unwrap_fns,
                manifold_unwrap_kwargs=manifold_unwrap_kwargs
            )

        # Set up metadata
        metadata = metadata or {}
        metadata.setdefault("height_threshold", height_threshold)
        metadata.setdefault("angle_indices", angle_indices)
        # Position indices for the torso/head height (typically z-coordinate)
        metadata.setdefault("height_index", 2)

        # Preprocessing functions depend on whether using manifold
        # - Manifold flow matching: manifold handles geometry natively (sphere for orientation)
        # - Euclidean (diffusion): needs unwrapping and augmentation for angles
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

        # Post-processing for inference (use index 0 for sphere coords)
        post_process_fns = [process_angles]
        post_process_fn_kwargs = {"angle_indices": [0]}

        # For manifold, set sphere indices to None in mins/maxs
        manifold_mins = mins.copy()
        manifold_maxs = maxs.copy()
        for idx in self.SPHERE_INDICES:
            if idx < len(manifold_mins):
                manifold_mins[idx] = None
                manifold_maxs[idx] = None

        super().__init__(
            name=name,
            state_dim=state_dim,
            stride=stride,
            history_length=history_length,
            horizon_length=horizon_length,
            max_path_length=max_path_length,
            mins=mins,
            maxs=maxs,
            angle_indices=angle_indices,
            manifold=manifold,
            manifold_mins=manifold_mins,
            manifold_maxs=manifold_maxs,
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

        Success if the humanoid has gotten up (height above threshold).
        """
        height_threshold = self.metadata.get("height_threshold", 0.8)
        height_index = self.metadata.get("height_index", 2)

        # Check if humanoid is upright based on height
        if len(state) > height_index:
            height = state[height_index]
            if height >= height_threshold:
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
        height_threshold = self.metadata.get("height_threshold", 0.8)
        height_index = self.metadata.get("height_index", 2)

        # Vectorized height check
        if states.shape[1] > height_index:
            heights = states[:, height_index]
            outcomes = np.where(
                heights >= height_threshold,
                Outcome.SUCCESS.value,
                Outcome.FAILURE.value
            )
        else:
            # All failures if state doesn't have height index
            outcomes = np.full(states.shape[0], Outcome.FAILURE.value)

        return outcomes.astype(np.int32)

    @classmethod
    def create(
        cls,
        stride: int = 1,
        history_length: int = 1,
        horizon_length: int = 31,
        max_path_length: Optional[int] = None,
        use_manifold: bool = False,
        **kwargs,
    ) -> "HumanoidGetUpSystem":
        """
        Factory method to create a HumanoidGetUpSystem with sensible defaults.

        Args:
            stride: Stride for trajectory sampling
            history_length: Length of history conditioning
            horizon_length: Length of prediction horizon
            max_path_length: Maximum trajectory length (uses default if None)
            use_manifold: Whether to use manifold-based flow matching
            **kwargs: Additional arguments passed to __init__

        Returns:
            HumanoidGetUpSystem instance
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
    ) -> "HumanoidGetUpSystem":
        """
        Create a HumanoidGetUpSystem from a config dictionary.

        This method extracts training parameters (stride, history_length, etc.)
        from the config while using system defaults for state space properties.

        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments to override config values

        Returns:
            HumanoidGetUpSystem instance
        """
        method_config = config.get("flow_matching", config.get("diffusion", {}))
        use_manifold = "manifold" in method_config and method_config.get("manifold") is not None

        return cls(
            name=kwargs.get("name", "humanoid_get_up"),
            stride=kwargs.get("stride", method_config.get("stride", 1)),
            history_length=kwargs.get(
                "history_length", method_config.get("history_length", 1)
            ),
            horizon_length=kwargs.get(
                "horizon_length", method_config.get("horizon_length", 31)
            ),
            max_path_length=kwargs.get("max_path_length"),
            height_threshold=kwargs.get("height_threshold", 0.8),
            use_manifold=kwargs.get("use_manifold", use_manifold),
            normalizer=kwargs.get("normalizer"),
            metadata=kwargs.get("metadata"),
        )

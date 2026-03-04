from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from genMoPlan.datasets.normalization import Normalizer
from genMoPlan.systems.base import BaseSystem, Outcome


def _create_quadrotor3d_manifold():
    """Create the manifold for 3D quadrotor (lazy import to avoid circular deps).

    State: [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r]
    Manifold: R^3 (position) x SO(3) (orientation via unit quaternion) x R^6 (velocities)
    """
    from flow_matching.utils.manifolds import SO3, Euclidean, Product

    return Product(
        input_dim=13,
        manifolds=[(Euclidean(), 3), (SO3(), 4, 3), (Euclidean(), 6)],
    )


def normalize_quaternions(data, quaternion_indices=None):
    """
    Post-processing function: normalize quaternion components to unit norm.

    Args:
        data: numpy array of shape (..., state_dim)
        quaternion_indices: list of 4 indices for [qw, qx, qy, qz]

    Returns:
        data with quaternion components normalized to unit norm
    """
    if quaternion_indices is None:
        raise ValueError("quaternion_indices must be provided")

    data = data.copy()
    q = data[..., quaternion_indices]
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    data[..., quaternion_indices] = q / norm
    return data


class Quadrotor3DLQRSystem(BaseSystem):
    """
    System descriptor for 3D Quadrotor LQR environment (PyBullet).

    State: [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r]
    - x, y, z: position (m)
    - qw, qx, qy, qz: unit quaternion orientation (scalar-first, canonicalized qw >= 0)
    - x_dot, y_dot, z_dot: linear velocities (m/s)
    - p, q, r: angular velocities in body frame (rad/s)

    Goal: Stabilize to hover at (x=0, y=0, z=1) with identity orientation.
    Success: Final state within Euclidean distance 0.05 of goal.
    """

    DEFAULT_STATE_DIM = 13
    DEFAULT_MINS = [
        -1.83, -1.83, 0.07, 0.0, -1.0, -1.0, -1.0,
        -3.22, -3.22, -3.32, -39.12, -38.87, -39.61,
    ]
    DEFAULT_MAXS = [
        1.83, 1.83, 3.03, 1.0, 1.0, 1.0, 1.0,
        3.22, 3.22, 3.0, 38.24, 39.11, 39.26,
    ]
    DEFAULT_MAX_PATH_LENGTH = 636
    DEFAULT_ANGLE_INDICES = []  # No simple angles; SO3 quaternions instead
    DEFAULT_STATE_NAMES = [
        "x", "y", "z", "qw", "qx", "qy", "qz",
        "x_dot", "y_dot", "z_dot", "p", "q", "r",
    ]

    # Quaternion indices in state vector
    QUATERNION_INDICES = [3, 4, 5, 6]

    # Goal state: hover at (0, 0, 1) with identity quaternion
    GOAL_STATE = np.array(
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    SUCCESS_RADIUS = 0.05

    # Failure limits from dataset termination thresholds
    FAILURE_LIMITS = np.array(
        [1.8, 1.8, np.inf, np.inf, np.inf, np.inf, np.inf,
         3.0, 3.0, 3.0, 24.0, 24.0, 24.0],
        dtype=np.float32,
    )
    # Z has asymmetric bounds
    Z_MIN = 0.1
    Z_MAX = 3.0

    def __init__(
        self,
        *,
        name: str = "quadrotor3d_lqr",
        dataset: str,
        state_dim: int = 13,
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
        success_radius: float = 0.05,
        use_manifold: bool = False,
    ):
        if max_path_length is None:
            max_path_length = self.DEFAULT_MAX_PATH_LENGTH
        if state_names is None:
            state_names = self.DEFAULT_STATE_NAMES.copy()
        if angle_indices is None:
            angle_indices = self.DEFAULT_ANGLE_INDICES.copy()
        if valid_outcomes is None:
            valid_outcomes = [Outcome.SUCCESS, Outcome.FAILURE, Outcome.INVALID]

        # Always create the TRUE manifold
        if manifold is None:
            from genMoPlan.utils.manifold import ManifoldWrapper

            raw_manifold = _create_quadrotor3d_manifold()
            manifold = ManifoldWrapper(raw_manifold)

        model_manifold = manifold if use_manifold else None

        metadata = metadata or {}
        metadata.setdefault("success_radius", success_radius)
        metadata.setdefault("goal_state", self.GOAL_STATE.tolist())
        metadata.setdefault("quaternion_indices", self.QUATERNION_INDICES)
        metadata.setdefault("failure_limits", self.FAILURE_LIMITS.tolist())
        metadata.setdefault("z_min", self.Z_MIN)
        metadata.setdefault("z_max", self.Z_MAX)
        metadata.setdefault("invalid_label", -1)
        metadata.setdefault("invalid_labels", [metadata["invalid_label"]])
        metadata.setdefault("invalid_outcomes", ["INVALID"])

        # Post-processing: normalize quaternions after generation
        post_process_fns = [normalize_quaternions]
        post_process_fn_kwargs = {"quaternion_indices": self.QUATERNION_INDICES}

        # Quaternion indices should NOT be normalized by the normalizer.
        # Override manifold_mins/maxs to set quaternion indices to None.
        manifold_mins = list(mins) if mins is not None else None
        manifold_maxs = list(maxs) if maxs is not None else None

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
            manifold_mins=manifold_mins,
            manifold_maxs=manifold_maxs,
            post_process_fns=post_process_fns,
            post_process_fn_kwargs=post_process_fn_kwargs,
            valid_outcomes=valid_outcomes,
            normalizer=normalizer,
            metadata=metadata,
        )

    @property
    def manifold_mins(self) -> List[Optional[float]]:
        """Quaternion indices are excluded from normalization (set to None)."""
        if self._manifold_mins is not None:
            return self._manifold_mins
        result = self.mins.copy()
        for idx in self.QUATERNION_INDICES:
            if idx < len(result):
                result[idx] = None
        return result

    @property
    def manifold_maxs(self) -> List[Optional[float]]:
        """Quaternion indices are excluded from normalization (set to None)."""
        if self._manifold_maxs is not None:
            return self._manifold_maxs
        result = self.maxs.copy()
        for idx in self.QUATERNION_INDICES:
            if idx < len(result):
                result[idx] = None
        return result

    def read_trajectory(self, sequence_path) -> np.ndarray:
        """
        Read a trajectory, normalizing quaternions to unit norm.

        No angle wrapping needed (quaternions, not Euler angles).
        """
        with open(sequence_path, "r") as f:
            lines = f.readlines()

        trajectory = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line == "":
                if i < len(lines) - 1:
                    raise ValueError(
                        f"[ {self.name} ] Empty line at {sequence_path} line {i}"
                    )
                else:
                    break

            state = line.split(",")
            state = [s for s in state if s != ""]

            if len(state) < self.state_dim:
                raise ValueError(
                    f"[ {self.name} ] Trajectory at {sequence_path} has {len(state)} "
                    f"states at line {i}, expected {self.state_dim}"
                )

            state = np.array([float(s) for s in state[: self.state_dim]], dtype=np.float32)

            # Normalize quaternion to unit norm
            q = state[self.QUATERNION_INDICES]
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-8:
                state[self.QUATERNION_INDICES] = q / q_norm

            trajectory.append(state)

        return np.array(trajectory, dtype=np.float32)

    def evaluate_final_state(self, state: np.ndarray) -> Outcome:
        state_vec = self._as_state_vector(state)
        outcomes = self.evaluate_final_states(state_vec[np.newaxis, :])
        return Outcome(outcomes[0])

    def evaluate_final_states(self, states: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of final states using vectorized, manifold-aware operations.

        Success: Within distance SUCCESS_RADIUS from GOAL_STATE.
        Failure: Out of bounds (position/velocity limits exceeded).
        """
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

        # Manifold-aware distance to goal
        states_t = torch.from_numpy(states).float()
        goal_t = torch.from_numpy(goal_state).float().expand_as(states_t)
        per_dim_dist = self.manifold.dist(states_t, goal_t)
        # per_dim_dist shape: (..., 10) = 3 (R^3) + 1 (SO3) + 6 (R^6)

        total_dist = torch.sqrt(torch.sum(per_dim_dist**2, dim=1))
        is_within_success = total_dist < success_radius

        # Check for out of bounds using distance from origin
        origin_t = torch.zeros_like(states_t)
        # For OOB check we need per-state-dim distances, but manifold dist
        # collapses SO3 to 1 dim. Use Euclidean dims directly for limit checks.
        # Position: indices 0-2
        # Velocities: indices 7-12
        euclidean_indices = [0, 1, 2, 7, 8, 9, 10, 11, 12]
        euclidean_limits_indices = [0, 1, 2, 7, 8, 9, 10, 11, 12]
        euclidean_states = states_t[:, euclidean_indices]
        euclidean_limits = torch.from_numpy(failure_limits[euclidean_limits_indices]).float()

        is_over_limit = torch.any(
            torch.abs(euclidean_states) > euclidean_limits, dim=1
        )

        # Z has asymmetric bounds
        z_values = states_t[:, 2]
        is_z_out = (z_values < z_min) | (z_values > z_max)

        is_out_of_bounds = is_over_limit | is_z_out

        outcomes = torch.full(
            (states.shape[0],), Outcome.INVALID.value, dtype=torch.int32
        )
        outcomes[is_within_success] = Outcome.SUCCESS.value
        outcomes[is_out_of_bounds] = Outcome.FAILURE.value

        return outcomes.detach().cpu().numpy()

    def _as_state_vector(self, state: np.ndarray) -> np.ndarray:
        state_np = np.asarray(state, dtype=np.float32)
        return state_np[0] if state_np.ndim > 1 else state_np

    @classmethod
    def create(
        cls,
        dataset: str,
        stride: int = 1,
        history_length: int = 1,
        horizon_length: int = 31,
        max_path_length: Optional[int] = None,
        use_manifold: bool = False,
        **kwargs,
    ) -> "Quadrotor3DLQRSystem":
        return cls(
            dataset=dataset,
            stride=stride,
            history_length=history_length,
            horizon_length=horizon_length,
            max_path_length=max_path_length,
            use_manifold=use_manifold,
            **kwargs,
        )

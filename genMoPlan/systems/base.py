import json
import pickle
from base64 import b64encode, b64decode
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from genMoPlan.datasets.normalization import Normalizer
from genMoPlan.utils.data_processing import compute_actual_length


class Outcome(IntEnum):
    """Canonical outcome labels with stable indices for probability vectors."""

    SUCCESS = 0
    FAILURE = 1
    INVALID = 2

    @property
    def index(self) -> int:
        return int(self.value)


class BaseSystem:
    """
    Dataset/system descriptor responsible for geometry, limits, normalization,
    rollout steps, and outcome evaluation.

    Subclasses should override hooks but reuse the caching/resolution helpers.

    System classes encapsulate all system-specific information:
    - State space: dimensions, names, limits
    - Geometry: manifold structure, angle indices
    - Data loading: trajectory reading, preprocessing
    - Inference: post-processing, manifold unwrapping
    - Evaluation: success/failure criteria
    """

    # Class-level defaults - subclasses should override these
    DEFAULT_STATE_DIM: int = 0
    DEFAULT_MINS: List[float] = []
    DEFAULT_MAXS: List[float] = []
    DEFAULT_MAX_PATH_LENGTH: int = 1000
    DEFAULT_ANGLE_INDICES: List[int] = []
    DEFAULT_STATE_NAMES: List[str] = []

    def __init__(
        self,
        *,
        name: str,
        state_dim: int,
        stride: int,
        history_length: int,
        horizon_length: int,
        max_path_length: int,
        # State space limits
        mins: Optional[List[float]] = None,
        maxs: Optional[List[float]] = None,
        state_names: Optional[List[str]] = None,
        angle_indices: Optional[List[int]] = None,
        # Manifold (optional)
        manifold: Optional[Any] = None,
        manifold_mins: Optional[List[Optional[float]]] = None,
        manifold_maxs: Optional[List[Optional[float]]] = None,
        # Data loading
        trajectory_preprocess_fns: Optional[List[Callable]] = None,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
        # Inference
        post_process_fns: Optional[List[Callable]] = None,
        post_process_fn_kwargs: Optional[Dict[str, Any]] = None,
        manifold_unwrap_fns: Optional[List[Callable]] = None,
        manifold_unwrap_kwargs: Optional[Dict[str, Any]] = None,
        # Outcomes and normalization
        valid_outcomes: Optional[Sequence[Outcome]] = None,
        normalizer: Optional[Normalizer] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        missing = []
        for key, val in [
            ("stride", stride),
            ("history_length", history_length),
            ("horizon_length", horizon_length),
            ("max_path_length", max_path_length),
        ]:
            if val is None:
                missing.append(key)
        if missing:
            raise ValueError(f"[ system ] Missing required step fields: {missing}")

        metadata = metadata or {}
        metadata.setdefault("invalid_label", -1)
        metadata.setdefault("invalid_labels", [metadata["invalid_label"]])
        # By default, only the INVALID outcome is treated as belonging to the
        # invalid class; concrete systems can override this in their metadata.
        metadata.setdefault("invalid_outcomes", ["INVALID"])

        self.name = name
        self.state_dim = int(state_dim)
        self.stride = int(stride)
        self.history_length = int(history_length)
        self.horizon_length = int(horizon_length)
        self.max_path_length = int(max_path_length)
        self.metadata = metadata

        # State space limits
        self.mins = mins if mins is not None else self.DEFAULT_MINS
        self.maxs = maxs if maxs is not None else self.DEFAULT_MAXS
        self.state_names = state_names if state_names is not None else self.DEFAULT_STATE_NAMES
        self.angle_indices = angle_indices if angle_indices is not None else self.DEFAULT_ANGLE_INDICES

        # Manifold structure (for flow matching on manifolds)
        self.manifold = manifold
        self._manifold_mins = manifold_mins
        self._manifold_maxs = manifold_maxs

        # Data loading functions
        self.trajectory_preprocess_fns = trajectory_preprocess_fns or []
        self.preprocess_kwargs = preprocess_kwargs or {}

        # Inference functions
        self.post_process_fns = post_process_fns or []
        self.post_process_fn_kwargs = post_process_fn_kwargs or {}
        self.manifold_unwrap_fns = manifold_unwrap_fns or []
        self.manifold_unwrap_kwargs = manifold_unwrap_kwargs or {}

        self.valid_outcomes: List[Outcome] = (
            list(valid_outcomes) if valid_outcomes else list(Outcome)
        )
        self._normalizer = normalizer

        # Cached values
        self._cached_num_inference_steps: Optional[int] = None
        self._num_inference_steps_override: Optional[int] = None

    # ------------------------------------------------------------------ #
    # State space properties
    # ------------------------------------------------------------------ #
    @property
    def manifold_mins(self) -> List[Optional[float]]:
        """
        Get mins for manifold normalization.
        Angle indices are set to None (not normalized in Euclidean space).
        """
        if self._manifold_mins is not None:
            return self._manifold_mins
        result = self.mins.copy()
        for idx in self.angle_indices:
            if idx < len(result):
                result[idx] = None
        return result

    @property
    def manifold_maxs(self) -> List[Optional[float]]:
        """
        Get maxs for manifold normalization.
        Angle indices are set to None (not normalized in Euclidean space).
        """
        if self._manifold_maxs is not None:
            return self._manifold_maxs
        result = self.maxs.copy()
        for idx in self.angle_indices:
            if idx < len(result):
                result[idx] = None
        return result

    # ------------------------------------------------------------------ #
    # Trajectory reading - subclasses should override
    # ------------------------------------------------------------------ #
    def read_trajectory(self, sequence_path: Union[str, Path]) -> np.ndarray:
        """
        Read a trajectory from a file path.

        This is a default implementation that reads CSV-like files.
        Subclasses can override for custom formats.

        Args:
            sequence_path: Path to trajectory file

        Returns:
            np.ndarray of shape (T, state_dim) with dtype float32
        """
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

            state = line.split(",")
            state = [s for s in state if s != ""]

            if len(state) < self.state_dim:
                raise ValueError(
                    f"[ {self.name} ] Trajectory at {sequence_path} has {len(state)} states at line {i}, expected {self.state_dim}"
                )

            state = state[: self.state_dim]
            state = np.array([float(s) for s in state])

            # Apply angle wrapping for angle indices
            for idx in self.angle_indices:
                if idx < len(state):
                    state[idx] = state[idx] % (2 * np.pi)  # wrap to [0, 2pi]
                    if state[idx] > np.pi:
                        state[idx] -= 2 * np.pi  # wrap to [-pi, pi]

            trajectory.append(state)

        return np.array(trajectory, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Configuration generation
    # ------------------------------------------------------------------ #
    def get_normalizer_params(self, use_manifold: bool = False) -> Dict[str, Any]:
        """
        Get normalizer parameters for the dataset.

        Args:
            use_manifold: If True, use manifold mins/maxs (None for angle indices)

        Returns:
            Dict with normalizer configuration
        """
        if use_manifold:
            return {
                "trajectory": {
                    "mins": self.manifold_mins,
                    "maxs": self.manifold_maxs,
                },
                "plan": None,
            }
        return {
            "trajectory": {
                "mins": self.mins,
                "maxs": self.maxs,
            },
            "plan": None,
        }

    def get_dataset_config(self, use_manifold: bool = False) -> Dict[str, Any]:
        """
        Get dataset configuration derived from system properties.

        Args:
            use_manifold: If True, use manifold-specific settings

        Returns:
            Dict with dataset configuration to merge with base config
        """
        config = {
            "observation_dim": self.state_dim,
            "angle_indices": self.angle_indices,
            "state_names": self.state_names,
            "max_path_length": self.max_path_length,
            "read_trajectory_fn": self.read_trajectory,
            "trajectory_normalizer": "LimitsNormalizer",
            "normalizer_params": self.get_normalizer_params(use_manifold=use_manifold),
            "trajectory_preprocess_fns": self.trajectory_preprocess_fns if not use_manifold else [],
            "preprocess_kwargs": self.preprocess_kwargs if not use_manifold else {},
        }

        # Add manifold if using manifold flow matching
        if use_manifold and self.manifold is not None:
            config["manifold"] = self.manifold

        return config

    def get_inference_config(self) -> Dict[str, Any]:
        """
        Get inference configuration derived from system properties.

        Returns:
            Dict with inference configuration
        """
        config = {
            "max_path_length": self.max_path_length,
            "post_process_fns": self.post_process_fns,
            "post_process_fn_kwargs": self.post_process_fn_kwargs,
            "manifold_unwrap_fns": self.manifold_unwrap_fns,
            "manifold_unwrap_kwargs": self.manifold_unwrap_kwargs,
        }
        return config

    def get_flow_matching_config(self) -> Dict[str, Any]:
        """
        Get flow matching specific configuration.

        Returns:
            Dict with flow matching configuration
        """
        config = {
            "manifold": self.manifold,
            "normalizer_params": self.get_normalizer_params(use_manifold=True),
            "trajectory_preprocess_fns": [],  # Manifold flow matching typically skips preprocessing
            "preprocess_kwargs": {},
            "manifold_unwrap_fns": self.manifold_unwrap_fns,
            "manifold_unwrap_kwargs": self.manifold_unwrap_kwargs,
        }
        return config

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #
    @property
    def normalizer(self) -> Optional[Normalizer]:
        return self._normalizer

    # ------------------------------------------------------------------ #
    # Outcome helpers
    # ------------------------------------------------------------------ #
    def outcome_to_index(self, outcome: Outcome) -> int:
        if outcome not in self.valid_outcomes:
            raise ValueError(f"[ system ] Outcome {outcome} not valid for {self.name}")
        return outcome.index

    # ------------------------------------------------------------------ #
    # Step / length resolution
    # ------------------------------------------------------------------ #
    def _compute_num_inference_steps(
        self,
        *,
        max_path_length: Optional[int] = None,
        horizon_length: Optional[int] = None,
        history_length: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> int:
        """Compute and cache num_inference_steps."""
        hist = int(history_length or self.history_length)
        horz = int(horizon_length or self.horizon_length)
        strd = int(stride or self.stride)
        max_len = int(max_path_length or self.max_path_length)

        actual_hist = compute_actual_length(hist, strd)
        actual_horz = compute_actual_length(horz, strd)
        actual_horz = max(actual_horz, 1)
        remaining = max(0, max_len - actual_hist)
        steps = int(np.ceil(remaining / actual_horz)) or 1
        self._cached_num_inference_steps = steps
        return steps

    def set_num_inference_steps_override(self, num_steps: int):
        """Force num_inference_steps and adjust max_path_length accordingly."""
        self._num_inference_steps_override = int(num_steps)
        actual_hist = compute_actual_length(self.history_length, self.stride)
        actual_horz = compute_actual_length(self.horizon_length, self.stride)
        actual_horz = max(actual_horz, 1)
        self.max_path_length = (
            actual_hist + self._num_inference_steps_override * actual_horz
        )
        self._cached_num_inference_steps = self._num_inference_steps_override

    def resolve_step_params(
        self,
        *,
        max_path_length: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        horizon_length: Optional[int] = None,
        history_length: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        """Return (horizon_length, max_path_length, num_inference_steps) using cache/override."""
        if horizon_length is not None:
            self.horizon_length = int(horizon_length)
        if history_length is not None:
            self.history_length = int(history_length)
        if stride is not None:
            self.stride = int(stride)
        if max_path_length is not None:
            self.max_path_length = int(max_path_length)

        if num_inference_steps is not None:
            self.set_num_inference_steps_override(num_inference_steps)
        elif self._cached_num_inference_steps is None:
            self._compute_num_inference_steps()

        resolved_steps = (
            self._num_inference_steps_override
            if self._num_inference_steps_override is not None
            else self._cached_num_inference_steps
        )
        assert resolved_steps is not None
        return self.horizon_length, self.max_path_length, resolved_steps

    # ------------------------------------------------------------------ #
    # Hooks to override
    # ------------------------------------------------------------------ #
    def should_terminate(
        self, state: np.ndarray, t: int, traj_so_far: Optional[np.ndarray]
    ):
        """Return Outcome to terminate or None to continue."""
        return None

    def evaluate_final_state(self, state: np.ndarray) -> Outcome:
        raise NotImplementedError

    def evaluate_final_states(self, states: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of final states using vectorized operations.

        Subclasses must implement this method with proper vectorization.
        Do NOT use Python loops - use NumPy vectorized operations.

        Args:
            states: Array of shape (batch_size, state_dim) containing final states

        Returns:
            np.ndarray of shape (batch_size,) with dtype int32 containing Outcome values
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement evaluate_final_states with vectorized operations"
        )

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    def _serialize_complex(self, obj: Any):
        try:
            json.dumps(obj)
            return {"kind": "json", "value": obj}
        except Exception:
            return {
                "kind": "pickle",
                "value": b64encode(pickle.dumps(obj)).decode("utf-8"),
            }

    def _deserialize_complex(self, payload: Dict[str, Any]):
        if payload["kind"] == "json":
            return payload["value"]
        return pickle.loads(b64decode(payload["value"].encode("utf-8")))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize system to dictionary.

        Note: Functions (preprocess_fns, post_process_fns, etc.) and manifolds
        are not serialized as they should be provided by the subclass when
        reconstructing from dict.
        """
        return {
            "name": self.name,
            "state_dim": self.state_dim,
            "stride": self.stride,
            "history_length": self.history_length,
            "horizon_length": self.horizon_length,
            "max_path_length": self.max_path_length,
            "mins": self.mins,
            "maxs": self.maxs,
            "state_names": self.state_names,
            "angle_indices": self.angle_indices,
            "valid_outcomes": [o.name for o in self.valid_outcomes],
            "metadata": self.metadata,
            "normalizer": (
                self._serialize_complex(self._normalizer) if self._normalizer else None
            ),
            "_num_inference_steps_override": self._num_inference_steps_override,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Reconstruct system from dictionary.

        Note: This uses the class defaults for functions and manifolds.
        Subclasses should override if they need custom behavior.
        """
        normalizer = None
        if data.get("normalizer"):
            normalizer = cls._deserialize_static(data["normalizer"])
        valid_outcomes = [Outcome[o] for o in data.get("valid_outcomes", [])]
        instance = cls(
            name=data["name"],
            state_dim=data["state_dim"],
            stride=data["stride"],
            history_length=data["history_length"],
            horizon_length=data["horizon_length"],
            max_path_length=data["max_path_length"],
            mins=data.get("mins"),
            maxs=data.get("maxs"),
            state_names=data.get("state_names"),
            angle_indices=data.get("angle_indices"),
            valid_outcomes=valid_outcomes,
            normalizer=normalizer,
            metadata=data.get("metadata", {}),
        )
        if data.get("_num_inference_steps_override") is not None:
            instance.set_num_inference_steps_override(
                data["_num_inference_steps_override"]
            )
        return instance

    @staticmethod
    def _deserialize_static(payload: Dict[str, Any]):
        if payload["kind"] == "json":
            return payload["value"]
        return pickle.loads(b64decode(payload["value"].encode("utf-8")))

import gc
import glob
import json
import multiprocessing
import os
import time
import warnings
from os import path
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from genMoPlan.datasets.normalization import get_normalizer, Normalizer
from genMoPlan.datasets.constants import MASK_ON, MASK_OFF
from genMoPlan.utils.arrays import to_torch
from genMoPlan.utils.data_processing import compute_actual_length
from genMoPlan.utils.model import load_model, get_normalizer_params
from genMoPlan.utils.params import load_inference_params
from genMoPlan.utils.progress import ETAIterator
from genMoPlan.utils.trajectory import plot_trajectories

# Inference mask strategy constants
INFERENCE_MASK_FIRST_STEP_ONLY = "first_step_only"
INFERENCE_MASK_ALWAYS = "always"
INFERENCE_MASK_NEVER = "never"
VALID_INFERENCE_MASK_STRATEGIES = [
    INFERENCE_MASK_FIRST_STEP_ONLY,
    INFERENCE_MASK_ALWAYS,
    INFERENCE_MASK_NEVER,
]
DEFAULT_INFERENCE_MASK_STRATEGY = INFERENCE_MASK_FIRST_STEP_ONLY

if TYPE_CHECKING:
    from genMoPlan.systems import BaseSystem, Outcome


def _infer_method_name(model_args) -> Optional[str]:
    if hasattr(model_args, "method_name"):
        return model_args.method_name
    method = getattr(model_args, "method", None)
    if method == "models.generative.Diffusion":
        return "diffusion"
    if method == "models.generative.FlowMatching":
        return "flow_matching"
    return None


def _generate_timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _convert_to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _convert_to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_json_safe(item) for item in obj]
    if callable(obj):
        return obj.__name__
    return obj


def _read_final_state_file(file_path: str) -> Optional[np.ndarray]:
    final_states = []
    start_dim = 0
    final_dim = 0
    final_start = None
    has_label_column = False

    try:
        with open(file_path, "r") as f:
            for line in f:
                line_data = line.strip().split(" ")
                if "label" in line_data:
                    has_label_column = True
                    continue

                if line.startswith("#") or line.startswith("s"):
                    for value in line_data:
                        if value.startswith("start"):
                            start_dim += 1
                        elif value.startswith("final"):
                            final_dim += 1

                    final_start = start_dim + 1 if has_label_column else start_dim
                    continue

                final_states.append(
                    [
                        np.float32(value)
                        for value in line_data[final_start : final_start + final_dim]
                    ]
                )

        return np.array(final_states, dtype=np.float32)
    except Exception as exc:
        print(
            f"[ utils/trajectory_generator ] Error loading final states from {file_path}: {exc}"
        )
        return None


def _latest_timestamp(model_path: str, final_state_directory: str) -> Optional[str]:
    target = path.join(model_path, final_state_directory)
    if not path.exists(target):
        return None

    candidates = [
        path.basename(d) for d in glob.glob(path.join(target, "*")) if path.isdir(d)
    ]
    if not candidates:
        return None

    return max(candidates)


class TrajectoryGenerator:
    """
    High-level helper for generating, storing, loading, and plotting trajectories.
    """

    def __init__(
        self,
        *,
        dataset: Optional[str],
        model_path: Optional[str] = None,
        model_state_name: str = "best.pt",
        model: Optional[torch.nn.Module] = None,
        model_args: Optional[object] = None,
        inference_params: Optional[dict] = None,
        device: str = "cuda",
        verbose: bool = True,
        default_batch_size: Optional[int] = None,
        system: Optional["BaseSystem"] = None,
    ):
        if inference_params is None:
            if dataset is None:
                raise ValueError(
                    "`dataset` must be provided when inference_params is None."
                )
            inference_params = load_inference_params(dataset)

        self.system = system
        self.dataset = dataset
        self.model_path = model_path
        self.model_state_name = model_state_name
        self.device = device
        self.verbose = verbose
        self.inference_params = inference_params
        self.model = None
        self.model_args = None
        self.method_name = None
        self.history_length = None
        self.horizon_length = None
        self.max_path_length = inference_params.get("max_path_length")
        self.stride = 1
        self.batch_size = default_batch_size or inference_params.get("batch_size")
        self.conditional_sample_kwargs = dict(
            inference_params.get("conditional_sample_kwargs", {})
        )
        self.post_process_fns = inference_params.get("post_process_fns")
        self.post_process_fn_kwargs = inference_params.get("post_process_fn_kwargs", {})
        self.final_state_directory = inference_params.get("final_state_directory")

        # Inference masking strategy
        self.inference_mask_strategy = inference_params.get(
            "inference_mask_strategy", DEFAULT_INFERENCE_MASK_STRATEGY
        )
        if self.inference_mask_strategy not in VALID_INFERENCE_MASK_STRATEGIES:
            raise ValueError(
                f"Invalid inference_mask_strategy '{self.inference_mask_strategy}'. "
                f"Valid options: {VALID_INFERENCE_MASK_STRATEGIES}"
            )

        # Derived state
        self._timestamp: Optional[str] = None
        self.gen_traj_path: Optional[str] = None
        self._last_final_states: Optional[np.ndarray] = None
        self._last_trajectories: Optional[np.ndarray] = None
        self._cached_num_inference_steps: Optional[int] = None

        self._load_model(model=model, model_args=model_args)
        self._initialize_method_state()
        self._ensure_system()
        self._apply_system_overrides()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _load_model(
        self, *, model: Optional[torch.nn.Module], model_args: Optional[object]
    ):
        if model is not None:
            if model_args is None:
                raise ValueError(
                    "`model_args` must be provided when supplying a pre-loaded model."
                )
            self.model = model
            self.model_args = model_args
            return

        if self.model_path is None:
            raise ValueError("`model_path` is required when model is not provided.")

        load_ema = self.inference_params.get("load_ema", False)
        self.model, self.model_args = load_model(
            self.model_path,
            self.device,
            self.model_state_name,
            verbose=self.verbose,
            inference_params=self.inference_params,
            load_ema=load_ema,
        )

    def _initialize_method_state(self):
        if self.model_args is None:
            raise ValueError("Model args were not loaded correctly.")

        self.method_name = _infer_method_name(self.model_args)
        self.history_length = int(getattr(self.model_args, "history_length"))
        self.horizon_length = int(getattr(self.model_args, "horizon_length"))
        self.stride = int(getattr(self.model_args, "stride", 1))

        if self.method_name and self.method_name in self.inference_params:
            self.conditional_sample_kwargs = dict(
                self.inference_params[self.method_name]
            )

    def _apply_system_overrides(self):
        if self.system is None:
            return
        self.history_length = int(self.system.history_length)
        self.horizon_length = int(self.system.horizon_length)
        self.stride = int(self.system.stride)
        if getattr(self.system, "max_path_length", None) is not None:
            self.max_path_length = int(self.system.max_path_length)
        # Keep inference params in sync for downstream consumers and saving
        self.inference_params["history_length"] = self.history_length
        self.inference_params["horizon_length"] = self.horizon_length
        self.inference_params["stride"] = self.stride
        self.inference_params["max_path_length"] = self.max_path_length
        if getattr(self.system, "_num_inference_steps_override", None) is not None:
            self._cached_num_inference_steps = self.system._num_inference_steps_override

    # ------------------------------------------------------------------
    # Normalization / processing helpers
    # ------------------------------------------------------------------
    def _get_normalizer(self) -> Optional[Normalizer]:
        if (
            self.system is not None
            and getattr(self.system, "normalizer", None) is not None
        ):
            return self.system.normalizer

        if self.model_args is None:
            return None

        normalizer_name = getattr(self.model_args, "trajectory_normalizer", None)
        if normalizer_name is None:
            return None

        params = get_normalizer_params(self.model_args)
        return get_normalizer(normalizer_name, params)

    @staticmethod
    def _process_states(
        states: np.ndarray,
        normalizer: Optional[Normalizer],
        post_process_fns: Optional[Sequence[Callable]],
        post_process_fn_kwargs: dict,
    ):
        if normalizer is not None:
            states = normalizer.unnormalize(states)

        if post_process_fns:
            for fn in post_process_fns:
                states = fn(states, **post_process_fn_kwargs)

        return states

    @staticmethod
    def _process_trajectories(
        trajectories: Sequence[np.ndarray],
        normalizer: Optional[Normalizer],
        post_process_fns: Optional[Sequence[Callable]],
        post_process_fn_kwargs: dict,
        verbose: bool,
    ):
        trajectory_list: List[np.ndarray] = (
            [trajectories]
            if isinstance(trajectories, np.ndarray)
            else list(trajectories)
        )

        if verbose:
            print("[ utils/trajectory_generator ] Processing trajectories")

        if normalizer is not None:
            concatenated = np.concatenate(trajectory_list, axis=0)
            concatenated = normalizer.unnormalize(concatenated)

            processed = []
            idx = 0
            for traj in trajectory_list:
                length = len(traj)
                processed.append(concatenated[idx : idx + length])
                idx += length

            trajectory_list = processed

        if post_process_fns:
            for fn in post_process_fns:
                trajectory_list = [
                    fn(traj, **post_process_fn_kwargs) for traj in trajectory_list
                ]

        return np.concatenate(trajectory_list, axis=0)

    # ------------------------------------------------------------------
    # Public single-run generation API
    # ------------------------------------------------------------------
    def generate_trajectories(
        self,
        unnormalized_start_states: np.ndarray,
        *,
        batch_size: Optional[int] = None,
        max_path_length: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        conditional_sample_kwargs: Optional[dict] = None,
        post_process_fns: Optional[Sequence[Callable]] = None,
        post_process_fn_kwargs: Optional[dict] = None,
        horizon_length: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Generates full trajectories for a single run using the loaded model.

        Args:
            unnormalized_start_states: Initial states (batch_size, observation_dim)
            batch_size: Batch size for generation
            max_path_length: Maximum trajectory length
            num_inference_steps: Number of inference steps (alternative to max_path_length)
            conditional_sample_kwargs: Additional kwargs for model sampling
            post_process_fns: Post-processing functions
            post_process_fn_kwargs: Kwargs for post-processing functions
            horizon_length: Horizon length override
            verbose: Verbosity flag

        Returns:
            np.ndarray: Full trajectories (batch_size, path_length, observation_dim)
        """
        return self._generate_trajectories_or_final_states(
            unnormalized_start_states,
            batch_size=batch_size,
            max_path_length=max_path_length,
            num_inference_steps=num_inference_steps,
            conditional_sample_kwargs=conditional_sample_kwargs,
            only_return_final_states=False,
            post_process_fns=post_process_fns,
            post_process_fn_kwargs=post_process_fn_kwargs,
            horizon_length=horizon_length,
            verbose=verbose,
        )

    def generate_final_states(
        self,
        unnormalized_start_states: np.ndarray,
        *,
        batch_size: Optional[int] = None,
        max_path_length: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        conditional_sample_kwargs: Optional[dict] = None,
        post_process_fns: Optional[Sequence[Callable]] = None,
        post_process_fn_kwargs: Optional[dict] = None,
        horizon_length: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Generates only final states for a single run using the loaded model.

        Args:
            unnormalized_start_states: Initial states (batch_size, observation_dim)
            batch_size: Batch size for generation
            max_path_length: Maximum trajectory length
            num_inference_steps: Number of inference steps (alternative to max_path_length)
            conditional_sample_kwargs: Additional kwargs for model sampling
            post_process_fns: Post-processing functions
            post_process_fn_kwargs: Kwargs for post-processing functions
            horizon_length: Horizon length override
            verbose: Verbosity flag

        Returns:
            np.ndarray: Final states only (batch_size, observation_dim)
        """
        return self._generate_trajectories_or_final_states(
            unnormalized_start_states,
            batch_size=batch_size,
            max_path_length=max_path_length,
            num_inference_steps=num_inference_steps,
            conditional_sample_kwargs=conditional_sample_kwargs,
            only_return_final_states=True,
            post_process_fns=post_process_fns,
            post_process_fn_kwargs=post_process_fn_kwargs,
            horizon_length=horizon_length,
            verbose=verbose,
        )

    def _generate_trajectories_or_final_states(
        self,
        unnormalized_start_states: np.ndarray,
        *,
        batch_size: Optional[int] = None,
        max_path_length: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        conditional_sample_kwargs: Optional[dict] = None,
        only_return_final_states: bool = False,
        post_process_fns: Optional[Sequence[Callable]] = None,
        post_process_fn_kwargs: Optional[dict] = None,
        horizon_length: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Internal method that generates trajectories or final states for a single run.
        """
        if (
            max_path_length is None
            and num_inference_steps is None
            and self.system is None
            and self.max_path_length is None
        ):
            raise ValueError(
                "Provide `max_path_length` or `num_inference_steps`, or supply a system."
            )

        if (
            max_path_length is not None
            and num_inference_steps is not None
            and self.system is None
        ):
            raise ValueError(
                "Provide only one of `max_path_length` or `num_inference_steps`."
            )

        horizon, resolved_max_path_length, resolved_num_steps = (
            self._resolve_step_params(
                max_path_length=max_path_length,
                num_inference_steps=num_inference_steps,
                horizon_length=horizon_length,
            )
        )
        verbose = self.verbose if verbose is None else verbose
        batch_size = int(
            batch_size or self.batch_size or len(unnormalized_start_states)
        )

        cond_kwargs = dict(self.conditional_sample_kwargs or {})
        if conditional_sample_kwargs:
            cond_kwargs.update(conditional_sample_kwargs)

        post_fns = (
            post_process_fns if post_process_fns is not None else self.post_process_fns
        )
        post_fn_kwargs = dict(self.post_process_fn_kwargs or {})
        if post_process_fn_kwargs:
            post_fn_kwargs.update(post_process_fn_kwargs)

        normalizer = self._get_normalizer()
        if normalizer is not None:
            start_states = normalizer(unnormalized_start_states)
        else:
            start_states = np.array(unnormalized_start_states, copy=True)

        total = len(start_states)

        if only_return_final_states:
            final_states = np.zeros_like(start_states)
        else:
            trajectory_batches: List[np.ndarray] = []

        for idx in range(0, total, batch_size):
            batch_start_states = start_states[idx : idx + batch_size]

            batch_result = self._generate_raw_trajectories(
                batch_start_states,
                max_path_length=resolved_max_path_length,
                num_inference_steps=resolved_num_steps,
                conditional_sample_kwargs=cond_kwargs,
                only_return_final_states=only_return_final_states,
                verbose=verbose,
                horizon_length=horizon,
            )

            if only_return_final_states:
                final_states[idx : idx + len(batch_start_states)] = batch_result
            else:
                trajectory_batches.append(batch_result)

        if only_return_final_states:
            return self._process_states(
                final_states, normalizer, post_fns, post_fn_kwargs
            )

        return self._process_trajectories(
            trajectory_batches, normalizer, post_fns, post_fn_kwargs, verbose
        )

    def _ensure_system(self):
        """
        Build or load a system if not provided. Preference:
        1) Use existing self.system.
        2) Load from inference_params["system_state"] if present.
        3) Construct a BaseSystem from available params/model_args.
        """
        # Import here to avoid circular dependency
        from genMoPlan.systems import BaseSystem, Outcome

        if self.system is not None:
            return

        stride = self.inference_params.get(
            "stride", getattr(self.model_args, "stride", None)
        )
        hist = self.inference_params.get(
            "history_length", getattr(self.model_args, "history_length", None)
        )
        horz = self.inference_params.get(
            "horizon_length", getattr(self.model_args, "horizon_length", None)
        )
        max_len = self.inference_params.get("max_path_length", None)

        missing = [
            k
            for k, v in [
                ("stride", stride),
                ("history_length", hist),
                ("horizon_length", horz),
                ("max_path_length", max_len),
            ]
            if v is None
        ]
        if missing:
            raise ValueError(
                f"[ utils/trajectory_generator ] Cannot build system; missing fields: {missing}"
            )

        normalizer = self._get_normalizer()
        valid_outcomes_raw = self.inference_params.get("valid_outcomes")
        valid_outcomes = None
        if valid_outcomes_raw:
            try:
                valid_outcomes = [
                    Outcome[o] if isinstance(o, str) else Outcome(int(o))
                    for o in valid_outcomes_raw
                ]
            except Exception:
                valid_outcomes = None

        self.system = BaseSystem(
            name=self.dataset or "unknown",
            state_dim=getattr(self.model_args, "observation_dim", 0),
            stride=stride,
            history_length=hist,
            horizon_length=horz,
            max_path_length=max_len,
            valid_outcomes=valid_outcomes,
            normalizer=normalizer,
            metadata={"source": "auto-built"},
        )

        # The constructed system is used for this run only; a JSON snapshot is
        # written alongside inference params for reproducibility when saving.

    # ------------------------------------------------------------------
    # Public configuration setters
    # ------------------------------------------------------------------
    def set_batch_size(self, batch_size: int):
        self.batch_size = int(batch_size)
        self.inference_params["batch_size"] = self.batch_size

    def set_horizon_and_max_path_lengths(
        self,
        *,
        horizon_length: Optional[int] = None,
        max_path_length: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
    ):
        if horizon_length is not None:
            self.horizon_length = int(horizon_length)
            self.inference_params["horizon_length"] = self.horizon_length

        if max_path_length is None and num_inference_steps is None:
            return

        if max_path_length is not None and num_inference_steps is not None:
            raise ValueError(
                "Cannot set both `max_path_length` and `num_inference_steps`."
            )

        if max_path_length is not None:
            self.max_path_length = int(max_path_length)
        else:
            actual_hist = compute_actual_length(self.history_length, self.stride)
            actual_horz = compute_actual_length(self.horizon_length, self.stride)
            self.max_path_length = actual_hist + (num_inference_steps * actual_horz)

        self.inference_params["max_path_length"] = self.max_path_length

    # ------------------------------------------------------------------
    # Timestamp / paths
    # ------------------------------------------------------------------
    @property
    def timestamp(self) -> Optional[str]:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: str):
        if value is None:
            raise ValueError(
                "Timestamp cannot be None. Use `ensure_timestamp` to auto-create one."
            )
        self._set_timestamp(value)

    def _set_timestamp(self, value: str):
        if self.model_path is None or self.final_state_directory is None:
            raise ValueError(
                "Cannot manage timestamps without `model_path` and `final_state_directory`."
            )

        self._timestamp = value
        self.gen_traj_path = path.join(
            self.model_path, self.final_state_directory, self._timestamp
        )
        os.makedirs(self.gen_traj_path, exist_ok=True)

    def ensure_timestamp(self, value: Optional[str] = None):
        if self._timestamp is not None and value is None:
            return
        target = value or _generate_timestamp()
        self._set_timestamp(target)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _save_inference_params(
        self, save_path: str, *, n_runs: Optional[int], batch_size: Optional[int]
    ):
        if self.inference_params is None:
            return
        json_safe_params = _convert_to_json_safe(dict(self.inference_params))

        if n_runs is not None:
            json_safe_params["n_runs"] = int(n_runs)
        if batch_size is not None:
            json_safe_params["batch_size"] = int(batch_size)

        json_path = path.join(save_path, "inference_params.json")
        with open(json_path, "w") as f:
            json.dump(json_safe_params, f, indent=4)

        if self.verbose:
            print(
                f"[ utils/trajectory_generator ] Inference params saved to {json_path}"
            )

        # Save the system snapshot as a separate JSON file for auditability.
        if self.system is not None:
            try:
                system_state = self.system.to_dict()
                system_path = path.join(save_path, "system_state.json")
                with open(system_path, "w") as f:
                    json.dump(system_state, f, indent=4)
                if self.verbose:
                    print(
                        f"[ utils/trajectory_generator ] System state saved to {system_path}"
                    )
            except Exception as exc:
                if self.verbose:
                    warnings.warn(
                        f"[ utils/trajectory_generator ] Failed to save system_state.json: {exc}"
                    )

    def _prepare_save_directory(
        self, n_runs: Optional[int], batch_size: Optional[int], timestamp: Optional[str]
    ):
        if self.model_path is None or self.final_state_directory is None:
            raise ValueError(
                "Saving final states requires `model_path` and `final_state_directory`."
            )

        self.ensure_timestamp(timestamp)
        assert self.gen_traj_path is not None
        self._save_inference_params(
            self.gen_traj_path, n_runs=n_runs, batch_size=batch_size
        )

    def _save_single_run(
        self, run_idx: int, start_states: np.ndarray, final_states: np.ndarray
    ):
        assert self.gen_traj_path is not None
        file_path = path.join(self.gen_traj_path, f"final_states_run_{run_idx}.txt")

        with open(file_path, "w") as f:
            f.write("# ")
            for i in range(start_states.shape[1]):
                f.write(f"start_{i} ")
            for i in range(final_states.shape[1]):
                f.write(f"final_{i} ")
            f.write("\n")

            for idx in range(start_states.shape[0]):
                for value in start_states[idx]:
                    f.write(f"{value} ")

                for value in final_states[idx]:
                    f.write(f"{value} ")

                f.write("\n")

        if self.verbose:
            print(f"[ utils/trajectory_generator ] Final states saved in {file_path}")

    # ------------------------------------------------------------------
    # Core generation logic
    # ------------------------------------------------------------------
    def generate_multiple_runs(
        self,
        start_states: np.ndarray,
        *,
        n_runs: int,
        batch_size: Optional[int] = None,
        discard_trajectories: bool = True,
        save: bool = False,
        timestamp: Optional[str] = None,
        run_offset: int = 0,
        metadata_n_runs: Optional[int] = None,
        max_path_length: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        horizon_length: Optional[int] = None,
        conditional_sample_kwargs: Optional[dict] = None,
        post_process_fns: Optional[Sequence] = None,
        post_process_fn_kwargs: Optional[dict] = None,
        verbose: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generates trajectories or final states for multiple runs.

        Returns:
            Tuple of (final_states, trajectories) where:
            - final_states: (n_runs, num_states, observation_dim)
            - trajectories: (n_runs, num_states, path_length, observation_dim) or None if discarded
        """
        if n_runs <= 0:
            raise ValueError("`n_runs` must be positive.")

        batch_size = int(batch_size or self.batch_size or len(start_states))
        verbose = self.verbose if verbose is None else verbose

        if save:
            self._prepare_save_directory(
                n_runs=metadata_n_runs or n_runs,
                batch_size=batch_size,
                timestamp=timestamp,
            )

        final_state_runs: List[np.ndarray] = []
        trajectory_runs: List[np.ndarray] = [] if not discard_trajectories else None

        iterator = ETAIterator(iter(range(n_runs)), n_runs)
        for run_idx in iterator:
            if verbose:
                print(
                    f"[ utils/trajectory_generator ] Run {run_idx + 1}/{n_runs} "
                    f"(Remaining Time: {iterator.eta_formatted})"
                )

            # Use the core generation method
            results = self._generate_trajectories_or_final_states(
                start_states,
                batch_size=batch_size,
                max_path_length=max_path_length,
                num_inference_steps=num_inference_steps,
                conditional_sample_kwargs=conditional_sample_kwargs,
                only_return_final_states=discard_trajectories,
                post_process_fns=post_process_fns,
                post_process_fn_kwargs=post_process_fn_kwargs,
                horizon_length=horizon_length,
                verbose=verbose,
            )

            # Extract final states and trajectories
            if discard_trajectories:
                final_states = results
                full_trajs = None
            else:
                final_states = results[:, -1].copy()
                full_trajs = results

            final_state_runs.append(final_states)
            if trajectory_runs is not None and full_trajs is not None:
                trajectory_runs.append(full_trajs)

            if save:
                self._save_single_run(run_offset + run_idx, start_states, final_states)

            # Clean up after each run
            gc.collect()
            try:
                params = list(self.model.parameters())
                if (
                    params
                    and params[0].device.type == "cuda"
                    and hasattr(torch, "cuda")
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()
            except Exception as exc:
                if verbose:
                    warnings.warn(
                        f"[ utils/trajectory_generator ] Error clearing CUDA cache: {exc}"
                    )

        final_state_runs = np.array(final_state_runs, dtype=np.float32)
        trajectories_result = (
            np.array(trajectory_runs, dtype=np.float32)
            if trajectory_runs is not None
            else None
        )

        self._last_final_states = final_state_runs
        self._last_trajectories = trajectories_result

        return final_state_runs, trajectories_result

    def _compute_inference_steps(
        self, max_path_length: Optional[int], horizon_length: Optional[int]
    ) -> int:
        if max_path_length is None and horizon_length is None:
            raise ValueError(
                "Either `max_path_length` or `num_inference_steps` must be provided."
            )

        if max_path_length is None:
            raise ValueError(
                "`max_path_length` must be provided when `num_inference_steps` is not set."
            )

        actual_hist = compute_actual_length(self.history_length, self.stride)
        actual_horz = compute_actual_length(
            horizon_length or self.horizon_length, self.stride
        )
        actual_horz = max(actual_horz, 1)
        remaining = max(0, int(max_path_length) - int(actual_hist))
        return int(np.ceil(remaining / actual_horz)) or 1

    def _resolve_step_params(
        self,
        *,
        max_path_length: Optional[int],
        num_inference_steps: Optional[int],
        horizon_length: Optional[int],
    ) -> Tuple[int, int, int]:
        """
        Resolve horizon_length, max_path_length, and num_inference_steps with caching/system support.
        """
        resolved_horizon = int(horizon_length or self.horizon_length)

        if self.system is not None:
            resolved_horizon, resolved_max, resolved_num = (
                self.system.resolve_step_params(
                    max_path_length=max_path_length,
                    num_inference_steps=num_inference_steps,
                    horizon_length=horizon_length,
                    history_length=self.history_length,
                    stride=self.stride,
                )
            )
            self.horizon_length = resolved_horizon
            self.max_path_length = resolved_max
            self._cached_num_inference_steps = resolved_num
            return resolved_horizon, resolved_max, resolved_num

        if num_inference_steps is not None:
            resolved_num = int(num_inference_steps)
            actual_hist = compute_actual_length(self.history_length, self.stride)
            actual_horz = compute_actual_length(resolved_horizon, self.stride)
            actual_horz = max(actual_horz, 1)
            resolved_max = actual_hist + resolved_num * actual_horz
        else:
            resolved_max = int(max_path_length or self.max_path_length)
            if (
                self._cached_num_inference_steps is not None
                and resolved_max == self.max_path_length
                and resolved_horizon == self.horizon_length
            ):
                resolved_num = self._cached_num_inference_steps
            else:
                resolved_num = self._compute_inference_steps(
                    resolved_max, resolved_horizon
                )

        self.horizon_length = resolved_horizon
        self.max_path_length = resolved_max
        self._cached_num_inference_steps = resolved_num
        return resolved_horizon, resolved_max, resolved_num

    def _generate_raw_trajectories(
        self,
        start_states: np.ndarray,
        *,
        max_path_length: Optional[int],
        num_inference_steps: Optional[int],
        conditional_sample_kwargs: dict,
        only_return_final_states: bool,
        verbose: bool,
        horizon_length: Optional[int],
    ) -> np.ndarray:
        if self.model is None or self.model_args is None:
            raise ValueError(
                "Model and model args must be loaded before generating trajectories."
            )

        model = self.model
        model_args = self.model_args
        device = self.device

        history_length = int(self.history_length)
        horizon = int(horizon_length or self.horizon_length)
        training_prediction_length = int(
            history_length + int(model_args.horizon_length)
        )

        prediction_length = horizon
        resolved_num_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(self._cached_num_inference_steps or 0)
        )
        if resolved_num_steps <= 0:
            if max_path_length is None:
                raise ValueError("num_inference_steps could not be resolved.")
            resolved_num_steps = self._compute_inference_steps(max_path_length, horizon)

        if max_path_length is None:
            actual_hist = compute_actual_length(history_length, self.stride)
            actual_horz = compute_actual_length(prediction_length, self.stride)
            actual_horz = max(actual_horz, 1)
            max_path_length = actual_hist + resolved_num_steps * actual_horz
        else:
            max_path_length = int(max_path_length)

        batch_size = len(start_states)
        current_states = to_torch(start_states, dtype=torch.float32, device=device)

        if not only_return_final_states:
            trajectories = np.zeros(
                (batch_size, max_path_length, model_args.observation_dim)
            )
            trajectories[:, 0] = np.array(start_states)
        else:
            trajectories = None

        history_window = torch.zeros(
            (batch_size, history_length, model_args.observation_dim),
            dtype=current_states.dtype,
            device=current_states.device,
        )

        use_history_padding = bool(getattr(model_args, "use_history_padding", True))

        if use_history_padding:
            history_window = current_states.unsqueeze(1).repeat(1, history_length, 1)
            first_step_mask = None
        else:
            history_window[:, -1, :] = current_states
            # Create mask following the correct convention:
            # - MASK_OFF (1.0): Position is valid/present
            # - MASK_ON (0.0): Position is masked/missing
            first_step_mask = torch.full(
                (batch_size, training_prediction_length),
                MASK_OFF,  # Default: all positions valid
                dtype=torch.float32,
                device=current_states.device,
            )
            if history_length > 1:
                # Mark early history positions as masked (only last position is valid initially)
                first_step_mask[:, : history_length - 1] = MASK_ON

        current_idx = history_length

        with tqdm(total=resolved_num_steps, disable=not verbose) as pbar:
            first_step = True
            while current_idx < max_path_length:
                slice_path_length = min(
                    prediction_length, max_path_length - current_idx
                )

                conditions = {t: history_window[:, t, :] for t in range(history_length)}

                sample_kwargs = dict(conditional_sample_kwargs or {})

                # Apply mask based on inference_mask_strategy
                # - "first_step_only": Mask only on first step (default, current behavior)
                # - "always": Pass mask on all autoregressive steps
                # - "never": Never pass mask during inference
                if first_step_mask is not None:
                    if self.inference_mask_strategy == INFERENCE_MASK_FIRST_STEP_ONLY:
                        if first_step:
                            sample_kwargs["mask"] = first_step_mask
                    elif self.inference_mask_strategy == INFERENCE_MASK_ALWAYS:
                        # For subsequent steps, all history is valid (model's own predictions)
                        if first_step:
                            sample_kwargs["mask"] = first_step_mask
                        else:
                            # Create a full mask where all positions are valid
                            all_valid_mask = torch.full(
                                (batch_size, training_prediction_length),
                                MASK_OFF,
                                dtype=torch.float32,
                                device=current_states.device,
                            )
                            sample_kwargs["mask"] = all_valid_mask
                    # For INFERENCE_MASK_NEVER, we don't add mask to sample_kwargs

                sample = model(
                    cond=conditions,
                    verbose=False,
                    return_chain=False,
                    **sample_kwargs,
                )

                traj_pred = sample.trajectories[
                    :, history_length : history_length + slice_path_length
                ]

                if trajectories is not None:
                    trajectories[:, current_idx : current_idx + slice_path_length] = (
                        traj_pred.cpu().numpy()
                    )

                with torch.no_grad():
                    history_window = torch.cat([history_window, traj_pred], dim=1)[
                        :, -history_length:, :
                    ]

                current_states = traj_pred[:, -1]
                current_idx += slice_path_length
                first_step = False

                del traj_pred
                termination = self._check_system_termination(
                    current_states,
                    current_idx,
                    trajectories if trajectories is not None else None,
                )
                if termination is not None:
                    if trajectories is not None:
                        last_np = current_states.cpu().numpy()
                        trajectories[:, current_idx:, :] = last_np[:, None, :]
                    break

                if (
                    device == "cuda"
                    and hasattr(torch, "cuda")
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

                pbar.update(1)

        if only_return_final_states:
            return current_states.cpu().detach().numpy()

        return trajectories

    def _check_system_termination(
        self,
        current_states: torch.Tensor,
        current_idx: int,
        trajectories: Optional[np.ndarray],
    ):
        if self.system is None:
            return None
        try:
            state_np = current_states.detach().cpu().numpy()
            traj_np = (
                trajectories[:, :current_idx]
                if trajectories is not None and current_idx > 0
                else None
            )
            return self.system.should_terminate(state_np, current_idx, traj_np)
        except Exception as exc:
            if self.verbose:
                warnings.warn(
                    f"[ utils/trajectory_generator ] Termination check failed: {exc}"
                )
            return None

    # ------------------------------------------------------------------
    # Loading / Saving utilities
    # ------------------------------------------------------------------
    def save_final_states(
        self,
        start_states: np.ndarray,
        final_states: np.ndarray,
        *,
        timestamp: Optional[str] = None,
        metadata_n_runs: Optional[int] = None,
    ):
        if final_states.ndim != 3:
            raise ValueError("`final_states` must be a 3D array.")

        run_first = self._ensure_run_first(final_states, start_states)
        n_runs = run_first.shape[0]

        self._prepare_save_directory(
            n_runs=metadata_n_runs or n_runs,
            batch_size=self.batch_size,
            timestamp=timestamp,
        )

        for run_idx in range(n_runs):
            self._save_single_run(run_idx, start_states, run_first[run_idx])

        print(
            f"[ utils/trajectory_generator ] Final states saved in {self.gen_traj_path}"
        )

    def load_saved_final_states(
        self,
        *,
        expected_runs: Optional[int] = None,
        timestamp: Optional[str] = None,
        parallel: bool = True,
    ) -> Tuple[np.ndarray, int, str]:
        if self.model_path is None or self.final_state_directory is None:
            raise ValueError(
                "Cannot load final states without `model_path` and `final_state_directory`."
            )

        resolved_timestamp = timestamp or _latest_timestamp(
            self.model_path, self.final_state_directory
        )
        if resolved_timestamp is None:
            raise ValueError("No stored timestamps found to load final states.")

        self._set_timestamp(resolved_timestamp)

        all_states: List[np.ndarray] = []
        file_paths: List[str] = []

        if expected_runs is not None:
            for run_idx in range(expected_runs):
                fpath = path.join(self.gen_traj_path, f"final_states_run_{run_idx}.txt")
                if os.path.exists(fpath):
                    file_paths.append(fpath)
                else:
                    print(
                        f"[ utils/trajectory_generator ] Expected {expected_runs} runs but "
                        f"only found {len(file_paths)} files."
                    )
                    break
        else:
            pattern = path.join(self.gen_traj_path, "final_states_run_*.txt")
            file_paths = sorted(glob.glob(pattern))

        if not file_paths:
            raise ValueError(f"No final state files found under {self.gen_traj_path}")

        if parallel:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                iterator = pool.imap(_read_final_state_file, file_paths)
                results = list(
                    tqdm(iterator, total=len(file_paths), disable=not self.verbose)
                )
        else:
            iterable = (
                tqdm(file_paths, desc="Loading final states")
                if self.verbose
                else file_paths
            )
            results = [_read_final_state_file(fpath) for fpath in iterable]

        for result in results:
            if result is not None:
                all_states.append(result)

        if not all_states:
            raise ValueError("Failed to load any final states.")

        final_states = np.array(all_states, dtype=np.float32)
        self._last_final_states = final_states
        return final_states, len(all_states), resolved_timestamp

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_generated_trajectories(
        self,
        *,
        trajectories: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        comparison_trajectories: Optional[np.ndarray] = None,
        show_traj_ends: bool = False,
        verbose: Optional[bool] = None,
    ):
        verbose = self.verbose if verbose is None else verbose
        trajs = trajectories if trajectories is not None else self._last_trajectories

        if trajs is None:
            raise ValueError("No trajectories available to plot.")

        formatted = self._flatten_for_plotting(trajs)
        plot_trajectories(
            formatted,
            image_path=image_path,
            verbose=verbose,
            comparison_trajectories=comparison_trajectories,
            show_traj_ends=show_traj_ends,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_run_first(
        final_states: np.ndarray, start_states: np.ndarray
    ) -> np.ndarray:
        if final_states.shape[0] == start_states.shape[0]:
            return final_states.transpose(1, 0, 2)
        return final_states

    @staticmethod
    def _flatten_for_plotting(trajectories: np.ndarray) -> np.ndarray:
        if isinstance(trajectories, list):
            return np.concatenate(trajectories, axis=0)

        if trajectories.ndim == 4:
            num_states, n_runs, length, dim = trajectories.shape
            return trajectories.reshape(num_states * n_runs, length, dim)

        if trajectories.ndim == 3:
            return trajectories

        raise ValueError("Unsupported trajectory array shape for plotting.")

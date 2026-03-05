import gc
import glob
import json
import multiprocessing
import os
import time
import warnings
from dataclasses import dataclass
from os import path
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from genMoPlan.datasets.normalization import Normalizer
from genMoPlan.utils.constants import MASK_ON, MASK_OFF
from genMoPlan.utils.arrays import to_torch
from genMoPlan.utils.data_processing import (
    compute_actual_length,
    compute_num_inference_steps,
    compute_max_path_length,
    compute_total_predictions,
    warn_stride_horizon_length,
)
from genMoPlan.utils.generation_result import GenerationResult, TerminationResult
from genMoPlan.utils.model import load_model
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
    from genMoPlan.systems import BaseSystem


@dataclass
class ResolvedParams:
    """Resolved generation parameters computed at generation time."""

    history_length: int
    horizon_length: int
    stride: int
    max_path_length: int
    num_inference_steps: int
    batch_size: int
    state_dim: int
    model_sequence_length: int  # history_length + horizon_length as model expects


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

    Args:
        dataset: Dataset name (used to load inference_params if not provided)
        model_path: Path to saved model checkpoint directory
        model_state_name: Name of checkpoint file (default: "best.pt")
        model: Pre-loaded model (alternative to model_path)
        model_args: Model arguments (required if model is provided)
        inference_params: Inference parameters dict (loaded from dataset config if not provided)
        device: Device to run generation on
        verbose: Whether to print progress information
        default_batch_size: Default batch size for generation
        system: System instance for termination checking and normalization
    """

    def __init__(
        self,
        *,
        dataset: Optional[str] = None,
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
        if system is None:
            raise ValueError(
                "TrajectoryGenerator requires a `system` instance; system=None is no longer supported."
            )

        if inference_params is None:
            if dataset is None and model_path is None:
                raise ValueError(
                    "`dataset` or `model_path` must be provided when inference_params is None."
                )
            if dataset is not None:
                inference_params = load_inference_params(dataset, system=system)
            else:
                inference_params = {}

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

        # Generation parameters (mirrors system; set by _setup_params_from_source)
        self.history_length: Optional[int] = None
        self.horizon_length: Optional[int] = None
        self.max_path_length: Optional[int] = None
        self.stride: int = 1
        self._termination_enabled: bool = True

        self._load_model(model=model, model_args=model_args)
        self._initialize_method_state()
        self._setup_params_from_source()

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
            system=self.system,  # Pass system for reduced parameter passing
        )

    def _initialize_method_state(self):
        if self.model_args is None:
            raise ValueError("Model args were not loaded correctly.")

        self.method_name = _infer_method_name(self.model_args)
        # Ensure model args match system (single source of truth)
        hist = int(getattr(self.model_args, "history_length"))
        horz = int(getattr(self.model_args, "horizon_length"))
        strd = int(getattr(self.model_args, "stride", 1))
        if hist != int(self.system.history_length) or horz != int(self.system.horizon_length) or strd != int(self.system.stride):
            raise ValueError(
                "Mismatch between model_args step parameters and system. "
                f"model_args(history={hist}, horizon={horz}, stride={strd}) vs "
                f"system(history={self.system.history_length}, horizon={self.system.horizon_length}, stride={self.system.stride})."
            )

        if self.method_name and self.method_name in self.inference_params:
            self.conditional_sample_kwargs = dict(
                self.inference_params[self.method_name]
            )

    def _setup_params_from_source(self):
        """
        Set generation parameters from the system (single source of truth).
        """
        self.history_length = int(self.system.history_length)
        self.horizon_length = int(self.system.horizon_length)
        self.stride = int(self.system.stride)
        self.max_path_length = int(self.system.max_path_length)

        warn_stride_horizon_length(
            self.horizon_length, self.stride, context="TrajectoryGenerator (system)"
        )

        if self.verbose:
            print(
                f"[ utils/trajectory_generator ] Using system parameters: "
                f"history={self.history_length}, horizon={self.horizon_length}, "
                f"stride={self.stride}, max_path={self.max_path_length}"
            )

        # Keep inference params in sync for downstream consumers and saving
        self.inference_params["history_length"] = self.history_length
        self.inference_params["horizon_length"] = self.horizon_length
        self.inference_params["stride"] = self.stride
        self.inference_params["max_path_length"] = self.max_path_length

    # ------------------------------------------------------------------
    # Normalization / processing helpers
    # ------------------------------------------------------------------
    def _get_normalizer(self) -> Optional[Normalizer]:
        # System is always present and owns the normalizer
        normalizer = getattr(self.system, "normalizer", None)
        if normalizer is None:
            raise ValueError(
                "TrajectoryGenerator requires system.normalizer to be initialized."
            )
        return normalizer

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
    # Public generation API
    # ------------------------------------------------------------------
    def generate(
        self,
        start_states: Optional[np.ndarray] = None,
        *,
        start_histories: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        max_path_length: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        horizon_length: Optional[int] = None,
        return_trajectories: bool = False,
        conditional_sample_kwargs: Optional[dict] = None,
        post_process_fns: Optional[Sequence[Callable]] = None,
        post_process_fn_kwargs: Optional[dict] = None,
        verbose: Optional[bool] = None,
    ) -> GenerationResult:
        """
        Generate trajectories from start states or history windows.

        Args:
            start_states: Initial states [N, state_dim], unnormalized.
                Mutually exclusive with start_histories.
            start_histories: Full history windows [N, history_length, state_dim], unnormalized.
                Used for evaluation where we have exact history from dataset.
                Mutually exclusive with start_states.
            batch_size: Batch size for generation
            max_path_length: Maximum trajectory length (mutually exclusive with num_inference_steps)
            num_inference_steps: Number of autoregressive steps (mutually exclusive with max_path_length)
            horizon_length: Override horizon length
            return_trajectories: If True, include full trajectories in result (memory intensive)
            conditional_sample_kwargs: Additional kwargs for model sampling
            post_process_fns: Post-processing functions to apply
            post_process_fn_kwargs: Kwargs for post-processing functions
            verbose: Verbosity flag

        Returns:
            GenerationResult with final_states and optionally trajectories
        """
        # Validate input: exactly one of start_states or start_histories must be provided
        if start_states is None and start_histories is None:
            raise ValueError("Either start_states or start_histories must be provided.")
        if start_states is not None and start_histories is not None:
            raise ValueError("Cannot provide both start_states and start_histories.")

        # Determine number of samples and infer start_states if using histories
        if start_histories is not None:
            n_samples = len(start_histories)
            # Extract last state of each history for output sizing
            start_states = start_histories[:, -1, :]
        else:
            n_samples = len(start_states)

        verbose = self.verbose if verbose is None else verbose

        # Resolve parameters using the new centralized method
        params = self._resolve_params(
            batch_size=batch_size or n_samples,
            max_path_length=max_path_length,
            num_inference_steps=num_inference_steps,
            horizon_length=horizon_length,
        )

        # Merge conditional sample kwargs
        cond_kwargs = dict(self.conditional_sample_kwargs or {})
        if conditional_sample_kwargs:
            cond_kwargs.update(conditional_sample_kwargs)

        # Resolve post-processing
        post_fns = (
            post_process_fns if post_process_fns is not None else self.post_process_fns
        )
        post_fn_kwargs = dict(self.post_process_fn_kwargs or {})
        if post_process_fn_kwargs:
            post_fn_kwargs.update(post_process_fn_kwargs)

        # Get normalizer
        normalizer = self._get_normalizer()

        # Normalize start states
        if normalizer is not None:
            start_states_normalized = normalizer(start_states)
        else:
            start_states_normalized = np.array(start_states, copy=True)

        # Normalize start histories if provided
        start_histories_normalized: Optional[np.ndarray] = None
        if start_histories is not None:
            if normalizer is not None:
                start_histories_normalized = normalizer(start_histories)
            else:
                start_histories_normalized = np.array(start_histories, copy=True)

        total = len(start_states_normalized)

        # Allocate output arrays
        all_final_states = np.zeros((total, params.state_dim), dtype=np.float32)
        all_trajectories: Optional[List[np.ndarray]] = (
            [] if return_trajectories else None
        )
        all_termination_steps = np.full(total, -1, dtype=np.int32)
        all_termination_outcomes = np.full(total, -1, dtype=np.int32)

        # Process in batches
        for idx in range(0, total, params.batch_size):
            batch_start = start_states_normalized[idx : idx + params.batch_size]
            actual_batch_size = len(batch_start)

            # Get batch histories if provided
            batch_histories = None
            if start_histories_normalized is not None:
                batch_histories = start_histories_normalized[
                    idx : idx + params.batch_size
                ]

            # Generate this batch
            batch_result = self._generate_batch(
                batch_start,
                params=params,
                normalizer=normalizer,
                return_trajectories=return_trajectories,
                conditional_sample_kwargs=cond_kwargs,
                verbose=verbose,
                start_histories_normalized=batch_histories,
            )

            # Store results
            all_final_states[idx : idx + actual_batch_size] = batch_result.final_states
            if batch_result.termination_steps is not None:
                all_termination_steps[idx : idx + actual_batch_size] = (
                    batch_result.termination_steps
                )
            if batch_result.termination_outcomes is not None:
                all_termination_outcomes[idx : idx + actual_batch_size] = (
                    batch_result.termination_outcomes
                )
            if all_trajectories is not None and batch_result.trajectories is not None:
                all_trajectories.append(batch_result.trajectories)

        # Post-process final states
        processed_final_states = self._process_states(
            all_final_states, normalizer, post_fns, post_fn_kwargs
        )

        # Post-process trajectories if present
        processed_trajectories = None
        if all_trajectories:
            processed_trajectories = self._process_trajectories(
                all_trajectories, normalizer, post_fns, post_fn_kwargs, verbose
            )

        # Determine if we have termination info
        has_termination = (all_termination_steps >= 0).any()

        return GenerationResult(
            final_states=processed_final_states,
            trajectories=processed_trajectories,
            termination_steps=all_termination_steps if has_termination else None,
            termination_outcomes=all_termination_outcomes if has_termination else None,
        )

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
        """Set generation length parameters. Resolution happens at generation time."""
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
            # Convert num_inference_steps to max_path_length for storage
            self.max_path_length = compute_max_path_length(
                num_inference_steps,
                self.history_length,
                self.horizon_length,
                self.stride,
            )

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
    # Multiple runs generation
    # ------------------------------------------------------------------
    def generate_multiple_runs(
        self,
        start_states: np.ndarray,
        *,
        n_runs: int,
        batch_size: Optional[int] = None,
        return_trajectories: bool = False,
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
        Generate trajectories for multiple runs (for statistical analysis).

        Args:
            start_states: Initial states [num_states, state_dim], unnormalized
            n_runs: Number of runs to generate
            batch_size: Batch size for generation
            return_trajectories: If True, return full trajectories (memory intensive)
            save: If True, save final states to disk
            timestamp: Timestamp for saving (auto-generated if None)
            run_offset: Offset for run numbering when saving
            metadata_n_runs: Override n_runs in saved metadata
            max_path_length: Maximum trajectory length
            num_inference_steps: Number of autoregressive steps
            horizon_length: Override horizon length
            conditional_sample_kwargs: Kwargs for model sampling
            post_process_fns: Post-processing functions
            post_process_fn_kwargs: Kwargs for post-processing
            verbose: Verbosity flag

        Returns:
            Tuple of (final_states, trajectories) where:
            - final_states: [n_runs, num_states, state_dim]
            - trajectories: [n_runs, num_states, path_length, state_dim] or None
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
        trajectory_runs: Optional[List[np.ndarray]] = (
            [] if return_trajectories else None
        )

        iterator = ETAIterator(iter(range(n_runs)), n_runs)
        for run_idx in iterator:
            if verbose:
                print(
                    f"[ utils/trajectory_generator ] Run {run_idx + 1}/{n_runs} "
                    f"(Remaining Time: {iterator.eta_formatted})"
                )

            # Use the new generate() method
            result = self.generate(
                start_states,
                batch_size=batch_size,
                max_path_length=max_path_length,
                num_inference_steps=num_inference_steps,
                horizon_length=horizon_length,
                return_trajectories=return_trajectories,
                conditional_sample_kwargs=conditional_sample_kwargs,
                post_process_fns=post_process_fns,
                post_process_fn_kwargs=post_process_fn_kwargs,
                verbose=verbose,
            )

            final_state_runs.append(result.final_states)
            if trajectory_runs is not None and result.trajectories is not None:
                trajectory_runs.append(result.trajectories)

            if save:
                self._save_single_run(
                    run_offset + run_idx, start_states, result.final_states
                )

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

        final_state_runs_arr = np.array(final_state_runs, dtype=np.float32)
        trajectories_result = (
            np.array(trajectory_runs, dtype=np.float32)
            if trajectory_runs is not None
            else None
        )

        return final_state_runs_arr, trajectories_result

    def _resolve_params(
        self,
        *,
        batch_size: Optional[int] = None,
        max_path_length: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        horizon_length: Optional[int] = None,
    ) -> ResolvedParams:
        """
        Resolve all generation parameters from config/overrides.

        Called once at start of generation to compute all necessary values.

        Args:
            batch_size: Override batch size
            max_path_length: Override max path length (mutually exclusive with num_inference_steps)
            num_inference_steps: Override number of inference steps
            horizon_length: Override horizon length

        Returns:
            ResolvedParams with all computed values
        """
        # Resolve horizon
        resolved_horizon = int(horizon_length or self.horizon_length)

        # Resolve max_path_length <-> num_inference_steps
        if num_inference_steps is not None and max_path_length is not None:
            raise ValueError(
                "Cannot specify both max_path_length and num_inference_steps."
            )

        if num_inference_steps is not None:
            resolved_num_steps = int(num_inference_steps)
            resolved_max_path = compute_max_path_length(
                resolved_num_steps, self.history_length, resolved_horizon, self.stride
            )
        elif max_path_length is not None:
            resolved_max_path = int(max_path_length)
            resolved_num_steps = compute_num_inference_steps(
                resolved_max_path, self.history_length, resolved_horizon, self.stride
            )
        elif self.max_path_length is not None:
            resolved_max_path = int(self.max_path_length)
            resolved_num_steps = compute_num_inference_steps(
                resolved_max_path, self.history_length, resolved_horizon, self.stride
            )
        else:
            raise ValueError(
                "Must provide max_path_length or num_inference_steps, "
                "or set max_path_length on the generator."
            )

        # Resolve batch size
        resolved_batch_size = int(batch_size or self.batch_size or 1)

        # Get state dimension from model_args
        state_dim = int(getattr(self.model_args, "observation_dim", 0))
        if state_dim == 0:
            raise ValueError("Could not determine state dimension from model_args.")

        # Model sequence length (what the model was trained on)
        model_sequence_length = int(
            self.history_length + self.model_args.horizon_length
        )

        return ResolvedParams(
            history_length=int(self.history_length),
            horizon_length=resolved_horizon,
            stride=int(self.stride),
            max_path_length=resolved_max_path,
            num_inference_steps=resolved_num_steps,
            batch_size=resolved_batch_size,
            state_dim=state_dim,
            model_sequence_length=model_sequence_length,
        )

    def _generate_batch(
        self,
        start_states_normalized: np.ndarray,
        *,
        params: ResolvedParams,
        normalizer: Optional[Normalizer],
        return_trajectories: bool,
        conditional_sample_kwargs: dict,
        verbose: bool,
        start_histories_normalized: Optional[np.ndarray] = None,
    ) -> GenerationResult:
        """
        Generate trajectories for a single batch.

        Args:
            start_states_normalized: Normalized initial states [batch, state_dim]
            start_states_unnormalized: Unnormalized initial states (for termination checking)
            params: Resolved generation parameters
            normalizer: Normalizer instance (for unnormalizing during termination check)
            return_trajectories: Whether to store full trajectories
            conditional_sample_kwargs: Kwargs for model sampling
            verbose: Verbosity flag
            start_histories_normalized: Optional normalized history windows [batch, history_length, state_dim]
            start_histories_unnormalized: Optional unnormalized history windows (for termination checking)

        Returns:
            GenerationResult for this batch
        """
        if self.model is None or self.model_args is None:
            raise ValueError(
                "Model and model args must be loaded before generating trajectories."
            )

        model = self.model
        device = self.device
        batch_size = len(start_states_normalized)

        # Convert to torch
        current_states = to_torch(
            start_states_normalized, dtype=torch.float32, device=device
        )

        # Initialize trajectory buffer if needed
        # Store trajectories in MODEL-SPACE (just model predictions) for efficiency
        # Shape: [batch, total_predictions, state_dim] where total_predictions includes history
        if return_trajectories:
            total_predictions = compute_total_predictions(
                params.history_length,
                params.num_inference_steps,
                params.horizon_length,
            )
            trajectories = np.zeros(
                (batch_size, total_predictions, params.state_dim),
                dtype=np.float32,
            )

            # Initialize history portion
            if start_histories_normalized is not None:
                # Copy provided history
                trajectories[:, : params.history_length] = start_histories_normalized
            else:
                # Store initial state at last history position
                trajectories[:, params.history_length - 1] = start_states_normalized
        else:
            trajectories = None

        # Initialize history window - use provided histories if available
        if start_histories_normalized is not None:
            history_window = to_torch(
                start_histories_normalized, dtype=torch.float32, device=device
            )
        else:
            history_window = self._init_history_window(current_states, params)

        # Initialize mask for first step if needed
        # When using exact histories, we don't need a mask (data is complete)
        if start_histories_normalized is not None:
            first_step_mask = None
        else:
            first_step_mask = self._init_first_step_mask(current_states, params)

        # Initialize termination tracking
        termination_result = TerminationResult.none(batch_size)
        terminated_mask = np.zeros(batch_size, dtype=bool)

        # Track current position in MODEL-SPACE (index into trajectory buffer)
        # This represents the number of model predictions made so far
        prediction_idx = params.history_length

        with tqdm(total=params.num_inference_steps, disable=not verbose) as pbar:
            first_step = True
            step_count = 0

            while step_count < params.num_inference_steps and not terminated_mask.all():
                # Compute how many steps to add this iteration
                steps_to_add = min(
                    params.horizon_length,
                    compute_total_predictions(
                        params.history_length,
                        params.num_inference_steps,
                        params.horizon_length,
                    )
                    - prediction_idx,
                )

                # Build conditions from history window
                conditions = {
                    t: history_window[:, t, :] for t in range(params.history_length)
                }

                # Build sample kwargs with mask handling
                sample_kwargs = self._build_sample_kwargs(
                    conditional_sample_kwargs,
                    first_step_mask,
                    first_step,
                    batch_size,
                    params,
                    current_states.device,
                )

                # Generate one horizon
                sample = model(
                    cond=conditions,
                    verbose=False,
                    return_chain=False,
                    **sample_kwargs,
                )

                # Extract predicted horizon (excluding history)
                horizon_pred = sample.trajectories[
                    :, params.history_length : params.history_length + steps_to_add
                ]

                # Store predictions in trajectory buffer (model-space indexing)
                if trajectories is not None:
                    trajectories[:, prediction_idx : prediction_idx + steps_to_add] = (
                        horizon_pred.cpu().numpy()
                    )

                # Check termination on ALL states in the predicted horizon
                # Pass model-space index so it can be converted to actual timesteps
                if self._termination_enabled:
                    new_termination = self._check_termination_all_states(
                        horizon_pred,
                        normalizer,
                        prediction_idx,
                        params.stride,
                        terminated_mask,
                    )
                    termination_result = termination_result.update(new_termination)
                    terminated_mask = termination_result.step >= 0

                # Update history window using rolling buffer
                history_window = self._update_history_window(
                    history_window, horizon_pred, params
                )

                # Update current states to last prediction
                current_states = horizon_pred[:, -1]
                prediction_idx += steps_to_add
                first_step = False
                step_count += 1

                # Memory cleanup
                del horizon_pred
                if (
                    device == "cuda"
                    and hasattr(torch, "cuda")
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

                pbar.update(1)

                # Early exit if all trajectories terminated
                if terminated_mask.all():
                    break

        # Build final states array
        final_states = current_states.cpu().detach().numpy()

        # Handle terminated trajectories - fill remaining model predictions with final state
        # Note: trajectories buffer is in MODEL-SPACE, but termination_step is in ACTUAL TIMESTEPS
        # We need to convert termination timesteps back to model prediction indices
        if trajectories is not None and terminated_mask.any():
            for i in range(batch_size):
                if termination_result.step[i] >= 0:
                    # Convert termination timestep to model prediction index
                    # termination_step[i] is actual timestep, need to find which prediction it corresponds to
                    term_timestep = termination_result.step[i]

                    # Convert to model prediction index (approximately)
                    # This is the inverse of: actual_timestep = compute_actual_length(model_idx, stride)
                    # For now, approximate as: model_idx ≈ term_timestep // stride
                    # More precise: need to account for history offset
                    if params.stride > 1:
                        # Rough approximation: convert back to model index
                        term_model_idx = (
                            term_timestep + params.stride - 1
                        ) // params.stride
                    else:
                        term_model_idx = term_timestep

                    # Ensure we don't go out of bounds
                    total_preds = compute_total_predictions(
                        params.history_length,
                        params.num_inference_steps,
                        params.horizon_length,
                    )
                    if term_model_idx < total_preds and term_model_idx > 0:
                        trajectories[i, term_model_idx:, :] = trajectories[
                            i, term_model_idx - 1, :
                        ]

        return GenerationResult(
            final_states=final_states,
            trajectories=trajectories,
            termination_steps=(
                termination_result.step if self._termination_enabled else None
            ),
            termination_outcomes=(
                termination_result.outcome if self._termination_enabled else None
            ),
        )

    def _init_history_window(
        self,
        current_states: torch.Tensor,
        params: ResolvedParams,
    ) -> torch.Tensor:
        """Initialize history window for autoregressive generation."""
        batch_size = current_states.shape[0]
        use_history_padding = bool(
            getattr(self.model_args, "use_history_padding", True)
        )

        if use_history_padding:
            # Repeat initial state across all history positions
            return current_states.unsqueeze(1).repeat(1, params.history_length, 1)
        else:
            # Initialize with zeros, place initial state at last position
            history_window = torch.zeros(
                (batch_size, params.history_length, params.state_dim),
                dtype=current_states.dtype,
                device=current_states.device,
            )
            history_window[:, -1, :] = current_states
            return history_window

    def _init_first_step_mask(
        self,
        current_states: torch.Tensor,
        params: ResolvedParams,
    ) -> Optional[torch.Tensor]:
        """Initialize mask for first step if history masking is used."""
        use_history_padding = bool(
            getattr(self.model_args, "use_history_padding", True)
        )

        if use_history_padding:
            return None

        batch_size = current_states.shape[0]
        # Create mask: MASK_OFF (1.0) = valid, MASK_ON (0.0) = masked
        first_step_mask = torch.full(
            (batch_size, params.model_sequence_length),
            MASK_OFF,
            dtype=torch.float32,
            device=current_states.device,
        )
        if params.history_length > 1:
            # Mark early history positions as masked (only last position is valid initially)
            first_step_mask[:, : params.history_length - 1] = MASK_ON

        return first_step_mask

    def _build_sample_kwargs(
        self,
        conditional_sample_kwargs: dict,
        first_step_mask: Optional[torch.Tensor],
        first_step: bool,
        batch_size: int,
        params: ResolvedParams,
        device: torch.device,
    ) -> dict:
        """Build kwargs for model sampling including mask handling."""
        sample_kwargs = dict(conditional_sample_kwargs or {})

        if first_step_mask is None:
            return sample_kwargs

        if self.inference_mask_strategy == INFERENCE_MASK_FIRST_STEP_ONLY:
            if first_step:
                sample_kwargs["mask"] = first_step_mask
        elif self.inference_mask_strategy == INFERENCE_MASK_ALWAYS:
            if first_step:
                sample_kwargs["mask"] = first_step_mask
            else:
                # All positions valid for subsequent steps
                sample_kwargs["mask"] = torch.full(
                    (batch_size, params.model_sequence_length),
                    MASK_OFF,
                    dtype=torch.float32,
                    device=device,
                )
        # For INFERENCE_MASK_NEVER, don't add mask

        return sample_kwargs

    def _update_history_window(
        self,
        history_window: torch.Tensor,
        horizon_pred: torch.Tensor,
        params: ResolvedParams,
    ) -> torch.Tensor:
        """Update history window with new predictions using efficient rolling."""
        with torch.no_grad():
            # Determine how many new states to use from horizon
            new_states_count = min(params.history_length, horizon_pred.shape[1])

            # Roll and update (more efficient than concat+slice for large tensors)
            if new_states_count >= params.history_length:
                # Replace entire history
                return horizon_pred[:, -params.history_length :, :].clone()
            else:
                # Partial update - roll left and fill from right
                history_window = torch.roll(history_window, -new_states_count, dims=1)
                history_window[:, -new_states_count:, :] = horizon_pred[
                    :, -new_states_count:, :
                ]
                return history_window

    def _check_termination_all_states(
        self,
        horizon_pred: torch.Tensor,
        normalizer: Optional[Normalizer],
        current_model_idx: int,
        stride: int,
        already_terminated: np.ndarray,
    ) -> TerminationResult:
        """
        Check termination for ALL states in the predicted horizon using vectorized evaluation.

        A trajectory terminates at the first state with SUCCESS or FAILURE outcome.
        Only continues if all states in the horizon are INVALID.

        Unnormalizes predictions before checking since evaluation logic operates on physical state values.

        Args:
            horizon_pred: Predicted horizon [batch, horizon_len, state_dim] (normalized)
            normalizer: Normalizer to unnormalize predictions
            current_model_idx: Current index in model-space (number of predictions made)
            stride: Temporal stride between predictions
            already_terminated: Boolean mask of already terminated trajectories

        Returns:
            TerminationResult with termination steps in ACTUAL TIMESTEPS (not model-space)
        """
        if normalizer is None:
            raise ValueError(
                "Termination checking requires a normalizer (expected: system.normalizer)."
            )

        batch_size, horizon_len, state_dim = horizon_pred.shape

        # Convert to numpy and unnormalize
        pred_np = horizon_pred.detach().cpu().numpy()
        # Unnormalize for evaluation
        pred_unnorm = normalizer.unnormalize(pred_np.reshape(-1, state_dim))
        pred_unnorm = pred_unnorm.reshape(batch_size, horizon_len, state_dim)

        # Vectorized evaluation: flatten all states and evaluate at once
        states_flat = pred_unnorm.reshape(
            -1, state_dim
        )  # [batch*horizon_len, state_dim]
        outcomes_flat = self.system.evaluate_final_states(
            states_flat
        )  # [batch*horizon_len]
        outcomes = outcomes_flat.reshape(
            batch_size, horizon_len
        )  # [batch, horizon_len]

        # Find first terminating state (SUCCESS or FAILURE) for each trajectory
        # Terminate if ANY state is SUCCESS/FAILURE, continue only if ALL are INVALID
        from genMoPlan.systems.base import Outcome

        is_terminating = (outcomes == Outcome.SUCCESS.value) | (
            outcomes == Outcome.FAILURE.value
        )

        termination_step = np.full(batch_size, -1, dtype=np.int32)
        termination_outcome = np.full(batch_size, -1, dtype=np.int32)

        # Convert current model index to actual timestep
        # current_model_idx is in model-space, need to convert to actual environment timesteps
        current_actual_timestep = compute_actual_length(current_model_idx, stride)

        for i in range(batch_size):
            # Skip if already terminated in previous horizon
            if already_terminated[i]:
                continue

            # Check if any state in this horizon triggers termination
            if is_terminating[i].any():
                # Find first terminating state in horizon (index in horizon)
                first_term_t = np.argmax(
                    is_terminating[i]
                )  # argmax returns index of first True

                # Convert to actual timestep:
                # Each prediction in horizon is stride timesteps apart
                # Termination happens at: current_actual_timestep + first_term_t * stride
                termination_step[i] = current_actual_timestep + first_term_t * stride
                termination_outcome[i] = outcomes[i, first_term_t]

        return TerminationResult(
            step=termination_step,  # In actual timesteps!
            outcome=termination_outcome,
            any_terminated=(termination_step >= 0).any(),
            all_terminated=(termination_step >= 0).all(),
        )

    # ------------------------------------------------------------------
    # Utility Functions
    # ------------------------------------------------------------------

    @staticmethod
    def convert_trajectory_to_timesteps(
        trajectory_model_space: np.ndarray,
        stride: int,
        max_timesteps: Optional[int] = None,
        fill_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convert model-space trajectory to actual timestep representation.

        Trajectories are stored densely in model-space (just the predictions).
        This converts them to a sparse representation at actual environment timesteps.

        Args:
            trajectory_model_space: [batch, num_predictions, state_dim] dense predictions
            stride: Temporal stride between predictions
            max_timesteps: Maximum timesteps to allocate (if None, computed from trajectory)
            fill_value: Value to fill non-prediction timesteps (default: None for zeros)

        Returns:
            trajectory_timesteps: [batch, max_timesteps, state_dim] sparse at strided indices

        Example:
            >>> # Model predicted 78 states with stride=10
            >>> traj_model = result.trajectories  # [batch, 78, 4]
            >>> traj_time = convert_trajectory_to_timesteps(traj_model, stride=10)
            >>> # Now [batch, 770, 4] with predictions at [0, 10, 20, ...]
        """
        batch_size, num_predictions, state_dim = trajectory_model_space.shape

        if max_timesteps is None:
            # Compute from stride: last prediction is at index (num_predictions-1) * stride
            max_timesteps = 1 + (num_predictions - 1) * stride

        # Allocate sparse buffer
        trajectory_timesteps = np.zeros(
            (batch_size, max_timesteps, state_dim),
            dtype=trajectory_model_space.dtype,
        )

        if fill_value is not None:
            trajectory_timesteps[:] = fill_value

        # Fill at strided positions
        for pred_idx in range(num_predictions):
            timestep_idx = pred_idx * stride
            if timestep_idx < max_timesteps:
                trajectory_timesteps[:, timestep_idx] = trajectory_model_space[
                    :, pred_idx
                ]

        return trajectory_timesteps

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
        return final_states, len(all_states), resolved_timestamp

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_generated_trajectories(
        self,
        trajectories: np.ndarray,
        *,
        image_path: Optional[str] = None,
        comparison_trajectories: Optional[np.ndarray] = None,
        show_traj_ends: bool = False,
        verbose: Optional[bool] = None,
    ):
        """
        Plot generated trajectories.

        Args:
            trajectories: Trajectories to plot (required)
            image_path: Path to save image
            comparison_trajectories: Optional comparison trajectories
            show_traj_ends: Whether to mark trajectory endpoints
            verbose: Verbosity flag
        """
        verbose = self.verbose if verbose is None else verbose
        trajs = trajectories

        if trajs is None:
            raise ValueError("No trajectories provided to plot.")

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

import os
from os import path
import json
import numpy as np
import warnings
import matplotlib.pyplot as plt
from typing import Any, Optional

from genMoPlan.utils import (
    JSONArgs,
    plot_trajectories,
    load_inference_params,
    load_roa_labels,
    compute_actual_length,
)
from genMoPlan.utils.systems import BaseSystem, Outcome
from genMoPlan.utils.trajectory_generator import TrajectoryGenerator


class Classifier:
    SUCCESS_COLOR_MAP = "tab10_r"
    FAILURE_COLOR_MAP = "tab20b"
    FP_COLOR_MAP = "bwr_r"
    FN_COLOR_MAP = "Wistia_r"
    SEPARATRIX_COLOR_MAP = "autumn_r"

    model_state_name: str = "best.pt"
    model_path: str = None
    results_path: str = None
    traj_plot_path: str = None
    method_name: str = None
    _trajectory_generator: TrajectoryGenerator = None

    dataset: str = None
    model_args: JSONArgs = None
    batch_size: int = None
    horizon_length: int = None
    history_length: int = None
    max_path_length: int = None
    conditional_sample_kwargs: dict = None

    start_states: np.ndarray = None
    expected_labels: np.ndarray = None
    n_runs: int = None
    _expected_n_runs: int = None
    trajectories: np.ndarray = None
    final_states: np.ndarray = None
    predicted_labels: np.ndarray = None
    uncertain_indices: np.ndarray = None
    separatrix_indices: np.ndarray = None

    # Classification
    inference_params: dict = None
    invalid_label: int = None
    invalid_labels: set = None
    invalid_outcome_indices: np.ndarray = None
    _outcome_prob_threshold: float = None
    labels_set: set = None
    labels_array: np.ndarray = None
    tp_mask: np.ndarray = None
    tn_mask: np.ndarray = None
    fp_mask: np.ndarray = None
    fn_mask: np.ndarray = None

    def __init__(
        self,
        dataset: str,
        model_state_name: str = "best.pt",
        model_path: str = None,
        n_runs: int = None,
        batch_size: int = None,
        verbose: bool = True,
        num_batches: int = None,
        device: str = "cuda",
        system: BaseSystem = None,
    ):
        self.dataset = dataset
        self.model_path = model_path
        self.verbose = verbose
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device
        self.system = system

        self._orig_n_runs = n_runs

        self.inference_params = load_inference_params(self.dataset)

        if model_path is not None:
            self._trajectory_generator = TrajectoryGenerator(
                dataset=self.dataset,
                model_path=self.model_path,
                model_state_name=model_state_name,
                inference_params=self.inference_params,
                device=self.device,
                verbose=self.verbose,
                default_batch_size=self.batch_size,
                system=self.system,
            )
            self.model_args = self.trajectory_generator.model_args
            self.method_name = self.trajectory_generator.method_name
            self.horizon_length = self.trajectory_generator.horizon_length
            self.history_length = self.trajectory_generator.history_length
            self.stride = self.trajectory_generator.stride
        else:
            self.method_name = None
            self.model_args = None
            self.horizon_length = None
            self.history_length = None
            self.stride = 1

        self._load_params()

    def _load_params(self):
        self.n_runs = (
            self._orig_n_runs
            if self._orig_n_runs is not None
            else self.inference_params["n_runs"]
        )
        self._expected_n_runs = self.n_runs
        self.batch_size = (
            self.batch_size
            if self.batch_size is not None
            else self.inference_params["batch_size"]
        )

        self._outcome_prob_threshold = getattr(self.system, "metadata", {}).get(
            "outcome_prob_threshold", None
        )
        if self._outcome_prob_threshold is None:
            self._outcome_prob_threshold = 0.0

        meta = getattr(self.system, "metadata", {}) or {}
        if "invalid_label" not in meta:
            raise ValueError("System metadata must define `invalid_label`.")
        self.invalid_label = meta["invalid_label"]
        self.invalid_labels = set(meta.get("invalid_labels", [self.invalid_label]))
        invalid_outcomes_spec = meta.get("invalid_outcomes", [])

        self.invalid_outcome_indices = self._resolve_invalid_outcome_indices(
            invalid_outcomes_spec
        )
        self.max_path_length = self.inference_params["max_path_length"]
        self.final_state_directory = self.inference_params["final_state_directory"]

        if self.trajectory_generator is not None:
            self.conditional_sample_kwargs = dict(
                self.trajectory_generator.conditional_sample_kwargs or {}
            )
        else:
            self.conditional_sample_kwargs = (
                self.inference_params[self.method_name]
                if self.method_name in self.inference_params
                else {}
            )
        self.ground_truth_filter_fn = (
            self.inference_params["ground_truth_filter_fn"]
            if "ground_truth_filter_fn" in self.inference_params
            else None
        )

        if self.system is None:
            raise ValueError(
                "Classifier requires a system; none was provided. "
                "Use the dataset config's `get_system()` when constructing the classifier."
            )

        # Outcome-based classification uses binary labels (0: outside RoA, 1: inside RoA)
        # plus the invalid label.
        self.labels_set = {0, 1}

        self.labels_array = np.array([*self.labels_set, self.invalid_label])

    @property
    def trajectory_generator(self) -> TrajectoryGenerator:
        if self._trajectory_generator is None:
            raise ValueError(
                "Trajectory generation requires `model_path` to be provided."
            )
        return self._trajectory_generator

    @property
    def timestamp(self):
        if self.trajectory_generator is None:
            return None
        return self.trajectory_generator.timestamp

    @timestamp.setter
    def timestamp(self, value):
        self.trajectory_generator.timestamp = value

    # === Helper methods ===

    def _setup_results_path(self):
        if self.results_path is not None:
            return

        if callable(self.inference_params["results_name"]):
            self.results_path = path.join(
                self.model_path,
                "results",
                self.inference_params["results_name"](
                    self.inference_params, self.method_name
                ),
            )
        else:
            self.results_path = path.join(
                self.model_path, "results", self.inference_params["results_name"]
            )

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self._save_inference_params()

        # Persist the system configuration alongside evaluation results for auditability.
        if self.system is not None:
            try:
                system_state = self.system.to_dict()
                system_path = path.join(self.results_path, "system_state.json")
                with open(system_path, "w") as f:
                    json.dump(system_state, f, indent=4)
            except Exception:
                if self.verbose:
                    print("[ utils/roa ] Failed to save system_state.json")

    def _setup_traj_plot_path(self):
        if self.traj_plot_path is not None:
            return

        if callable(self.inference_params["results_name"]):
            self.traj_plot_path = path.join(
                self.model_path,
                "viz_trajs",
                self.inference_params["results_name"](
                    self.inference_params, self.method_name
                ),
            )
        else:
            self.traj_plot_path = path.join(
                self.model_path, "viz_trajs", self.inference_params["results_name"]
            )

        if not os.path.exists(self.traj_plot_path):
            os.makedirs(self.traj_plot_path)

        self._save_inference_params(self.traj_plot_path)

    def _save_inference_params(self, save_path: str = None):
        if save_path is None:
            save_path = self.results_path

        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k): convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            elif callable(obj):
                return obj.__name__
            else:
                return obj

        json_safe_params = convert_keys(self.inference_params)

        json_safe_params["n_runs"] = self.n_runs
        json_safe_params["batch_size"] = self.batch_size

        json_fpath = path.join(save_path, "inference_params.json")

        with open(json_fpath, "w") as f:
            json.dump(json_safe_params, f, indent=4)

        if self.verbose:
            print(f"[ utils/roa ] Inference params saved in {json_fpath}")

    def _save_predicted_labels(self):
        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")

        self._setup_results_path()

        file_path = path.join(self.results_path, f"predicted_labels.txt")

        with open(file_path, "w") as f:
            f.write("# ")
            for i in range(self.start_states.shape[1]):
                f.write(f"start_{i} ")
            f.write("label\n")

            for start_point_idx in range(self.start_states.shape[0]):
                for i in range(self.start_states.shape[1]):
                    f.write(f"{self.start_states[start_point_idx, i]} ")

                f.write(f"{self.predicted_labels[start_point_idx]} ")

                f.write("\n")
        if self.verbose:
            print(f"[ utils/roa ] Saved predicted labels to {file_path}")

    # === Public Setters ===

    def set_outcome_prob_threshold(self, prob_threshold: float):
        """
        Set the probability threshold used when converting probabilities
        (over outcomes or labels) into hard decisions.
        """
        self._outcome_prob_threshold = prob_threshold
        self.inference_params["outcome_prob_threshold"] = prob_threshold

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        self.inference_params["batch_size"] = batch_size
        if self.trajectory_generator is not None:
            self.trajectory_generator.set_batch_size(batch_size)

    def set_horizon_and_max_path_lengths(
        self,
        horizon_length: Optional[int] = None,
        *,
        max_path_length: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
    ):
        if horizon_length is not None:
            self.horizon_length = horizon_length
            self.inference_params["horizon_length"] = horizon_length

        if self.trajectory_generator is not None:
            self.trajectory_generator.set_horizon_and_max_path_lengths(
                horizon_length=horizon_length,
                max_path_length=max_path_length,
                num_inference_steps=num_inference_steps,
            )
            self.max_path_length = self.trajectory_generator.max_path_length
            return

        if max_path_length is None and num_inference_steps is None:
            return

        if max_path_length is not None and num_inference_steps is not None:
            raise ValueError("Cannot set both max_path_length and num_inference_steps")

        if max_path_length is not None:
            self.max_path_length = max_path_length
        else:
            actual_hist = compute_actual_length(
                self.history_length, getattr(self, "stride", 1)
            )
            actual_horz = compute_actual_length(
                self.horizon_length, getattr(self, "stride", 1)
            )
            self.max_path_length = actual_hist + (num_inference_steps * actual_horz)

        self.inference_params["max_path_length"] = self.max_path_length

    def reset(self, for_analysis: bool = False):
        """
        Resets the classifier for new runs while retaining the ground truth data

        This method is useful when you want to perform a new analysis with different parameters
        on the same ground truth data without re-initializing the entire object.

        Args:
            for_analysis: If True, only the analysis results will be reset but the run data
                          (e.g., final_states, trajectories) will be retained.

        **Set/Modified attributes:**
            - `results_path`
            - `traj_plot_path`
            - `predicted_labels`
            - `uncertain_indices`
            - `separatrix_indices`
            - `tp_mask`, `tn_mask`, `fp_mask`, `fn_mask`
            - (if `for_analysis` is False): `final_states`, `trajectories`, `n_runs`
        """
        self.results_path = None
        self.traj_plot_path = None
        self._load_params()

        self.predicted_labels = None
        self.uncertain_indices = None
        self.separatrix_indices = None
        self.label_probabilities = None
        self.tp_mask = None
        self.tn_mask = None
        self.fp_mask = None
        self.fn_mask = None

        self.outcome_labels = None
        self.outcome_probabilities = None
        self.predicted_outcomes = None
        self.outcome_uncertain_indices = None

        if for_analysis:
            return

        self.final_states = None
        self.trajectories = None
        self.n_runs = self._orig_n_runs

    # === Plotters and Savers ===

    def save_final_states(self):
        """Saves the final states of each run to disk.

        **Required attributes:**
            - `final_states`
            - `start_states`
        """
        if self.verbose:
            print(f"[ utils/roa ] Saving final states")
        if self.final_states is None:
            raise ValueError("No final states available")
        if self.start_states is None:
            raise ValueError("Start states must be available to save final states.")

        generator = self.trajectory_generator
        final_states_run_first = self.final_states.transpose(1, 0, 2)
        generator.save_final_states(
            start_states=self.start_states,
            final_states=final_states_run_first,
            timestamp=self.timestamp,
            metadata_n_runs=self._expected_n_runs,
        )

    def plot_trajectories(self):
        """
        Plots the generated trajectories.

        **Required attributes:**
            - `trajectories`: Full trajectory data.
            - `start_states`: For validation and plot limits.
        """
        self._setup_traj_plot_path()

        if self.verbose:
            print(f"[ utils/roa ] Plotting trajectories")

        if self.trajectories is None:
            raise ValueError("No trajectories available")

        if self.start_states.shape[1] > 2:
            raise ValueError("Cannot plot trajectories for more than 2D start points")

        trajectories = np.concatenate(self.trajectories, axis=0)

        image_name = f"trajectories_{len(self.start_states)}_{self.n_runs}.png"
        image_path = path.join(self.traj_plot_path, image_name)

        plot_trajectories(trajectories, image_path, self.verbose)

        if self.verbose:
            print(f"[ utils/roa ] Trajectories plotted in {image_path}")

    # Legacy plotting helpers removed; plots now operate on outcome-based labels.

    def plot_roas(self, plot_separatrix=True, s=0.1):
        """
        Plots the estimated regions of attraction (ROAs).

        **Required attributes:**
            - `predicted_labels`
            - `start_states`
            - `labels_array`, `labels_set`
            - `separatrix_indices` (if `plot_separatrix=True`)
        """
        self._setup_results_path()

        if self.verbose:
            print(f"[ utils/roa ] Plotting ROAs")
        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")

        if self.start_states.shape[1] > 2:
            raise ValueError("Cannot plot ROAs for more than 2D start points")

        labels_to_plot = self.predicted_labels.copy()
        labels_to_plot[labels_to_plot == self.invalid_label] = len(self.labels_set)

        success_mask = labels_to_plot == self.labels_array[1]
        failure_mask = labels_to_plot == self.labels_array[0]
        separatrix_mask = labels_to_plot == len(self.labels_set)

        plt.figure(figsize=(10, 8))

        plt.scatter(
            self.start_states[success_mask, 0],
            self.start_states[success_mask, 1],
            c=labels_to_plot[success_mask],
            s=s,
            cmap=self.SUCCESS_COLOR_MAP,
        )
        plt.scatter(
            self.start_states[failure_mask, 0],
            self.start_states[failure_mask, 1],
            c=labels_to_plot[failure_mask],
            s=s,
            cmap=self.FAILURE_COLOR_MAP,
        )
        plt.scatter(
            self.start_states[separatrix_mask, 0],
            self.start_states[separatrix_mask, 1],
            c=labels_to_plot[separatrix_mask],
            s=s,
            cmap=self.SEPARATRIX_COLOR_MAP,
        )

        plt.title("RoAs")
        plt.savefig(path.join(self.results_path, f"roas.png"))
        plt.close()

        if plot_separatrix:
            plt.figure(figsize=(10, 8))
            plt.scatter(
                self.start_states[self.separatrix_indices, 0],
                self.start_states[self.separatrix_indices, 1],
                c="red",
                s=s,
            )
            plt.title("Separatrix")
            plt.savefig(path.join(self.results_path, f"separatrix.png"))
            plt.close()

    def plot_classification_results(self, s=0.1):
        """
        Plots the classification results, showing TP, TN, FP, FN points.

        **Required attributes:**
            - `predicted_labels`
            - `start_states`
            - `tp_mask`, `tn_mask`, `fp_mask`, `fn_mask`: Computed in `compute_classification_results`.
        """
        self._setup_results_path()

        if self.verbose:
            print(f"[ utils/roa ] Plotting classification results")

        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")

        if (
            self.tp_mask is None
            or self.tn_mask is None
            or self.fp_mask is None
            or self.fn_mask is None
        ):
            raise ValueError(
                "Classification results not computed. Call compute_prediction_metrics first."
            )

        if self.start_states.shape[1] > 2:
            raise ValueError(
                "Cannot plot classification results for more than 2D start points"
            )

        # Plot true positives
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.start_states[self.tp_mask, 0],
            self.start_states[self.tp_mask, 1],
            c=self.predicted_labels[self.tp_mask],
            s=s,
            cmap=self.SUCCESS_COLOR_MAP,
        )
        plt.title("True Positives")
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "true_positives.png"))
        plt.close()

        # Plot true negatives
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.start_states[self.tn_mask, 0],
            self.start_states[self.tn_mask, 1],
            c=self.predicted_labels[self.tn_mask],
            s=s,
            cmap=self.FAILURE_COLOR_MAP,
        )
        plt.title("True Negatives")
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "true_negatives.png"))
        plt.close()

        # Plot false positives
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.start_states[self.fp_mask, 0],
            self.start_states[self.fp_mask, 1],
            c=self.predicted_labels[self.fp_mask],
            s=s,
            cmap=self.FP_COLOR_MAP,
        )
        plt.title("False Positives")
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "false_positives.png"))
        plt.close()

        # Plot false negatives
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.start_states[self.fn_mask, 0],
            self.start_states[self.fn_mask, 1],
            c=self.predicted_labels[self.fn_mask],
            s=s,
            cmap=self.FN_COLOR_MAP,
        )
        plt.title("False Negatives")
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "false_negatives.png"))
        plt.close()

        # Combined plots with all incorrect classifications
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.start_states[self.fp_mask, 0],
            self.start_states[self.fp_mask, 1],
            c=self.predicted_labels[self.fp_mask],
            s=s,
            cmap=self.FP_COLOR_MAP,
            label="False Positives",
        )
        plt.scatter(
            self.start_states[self.fn_mask, 0],
            self.start_states[self.fn_mask, 1],
            c=self.predicted_labels[self.fn_mask],
            s=s,
            cmap=self.FN_COLOR_MAP,
            label="False Negatives",
        )
        plt.title("Incorrect Classifications")
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "incorrect_classifications.png"))
        plt.close()

        # Combined plot with all classifications
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.start_states[self.tp_mask, 0],
            self.start_states[self.tp_mask, 1],
            c=self.predicted_labels[self.tp_mask],
            s=s,
            cmap=self.SUCCESS_COLOR_MAP,
            label="True Positives",
        )
        plt.scatter(
            self.start_states[self.tn_mask, 0],
            self.start_states[self.tn_mask, 1],
            c=self.predicted_labels[self.tn_mask],
            s=s,
            cmap=self.FAILURE_COLOR_MAP,
            label="True Negatives",
        )
        plt.scatter(
            self.start_states[self.fp_mask, 0],
            self.start_states[self.fp_mask, 1],
            c=self.predicted_labels[self.fp_mask],
            s=s,
            cmap=self.FP_COLOR_MAP,
            label="False Positives",
        )
        plt.scatter(
            self.start_states[self.fn_mask, 0],
            self.start_states[self.fn_mask, 1],
            c=self.predicted_labels[self.fn_mask],
            s=s,
            cmap=self.FN_COLOR_MAP,
            label="False Negatives",
        )
        plt.scatter(
            self.start_states[self.separatrix_indices, 0],
            self.start_states[self.separatrix_indices, 1],
            c=self.predicted_labels[self.separatrix_indices],
            s=s,
            cmap=self.SEPARATRIX_COLOR_MAP,
            label="Separatrix",
        )
        plt.title("Classification Results")
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "classification_results.png"))
        plt.close()

        if self.verbose:
            print(
                f"[ utils/roa ] Classification result plots saved in {self.results_path}"
            )

    # === RoA Estimation Steps ===

    def load_ground_truth(self):
        """
        Loads ground truth start states and expected labels from a file.

        This is a necessary first step for any analysis that involves comparing
        model predictions to a ground truth.

        **Required attributes:**
            - `dataset`: Name of the dataset, used to find the `roa_labels.txt` file.

        **Set/Modified attributes:**
            - `start_states`: Loaded start states.
            - `expected_labels`: Loaded ground truth labels corresponding to start states.
            - `batch_size`: May be modified if `num_batches` is set.

        Raises:
            FileNotFoundError: If `roa_labels.txt` is not found for the dataset.
        """
        if self.verbose:
            print("[ scripts/estimate_roa ] Loading ground truth")

        start_states, expected_labels = load_roa_labels(self.dataset)

        if self.ground_truth_filter_fn is not None and self.model_args is not None:
            start_states, expected_labels = self.ground_truth_filter_fn(
                start_states, expected_labels, self.model_args
            )

        self.start_states = start_states
        self.expected_labels = expected_labels

        if self.num_batches is not None:
            self.set_batch_size(self.start_states.shape[0] // self.num_batches)

        if self.verbose:
            print(
                f"[ utils/roa ] Loaded {self.start_states.shape[0]} ground truth data points"
            )

    def generate_trajectories(
        self,
        discard_trajectories: bool = True,
        save: bool = False,
    ):
        """
        Generates trajectories from `start_states` using the loaded model.

        This method is one of the ways to populate `final_states`.
        """
        generator = self.trajectory_generator

        if self.start_states is None:
            raise ValueError(
                "Start states must be loaded before generating trajectories."
            )

        if self.n_runs is None:
            raise ValueError(
                "Number of runs (`n_runs`) must be defined before generation."
            )

        existing_runs = 0 if self.final_states is None else self.final_states.shape[1]
        runs_needed = self._expected_n_runs - existing_runs

        if runs_needed <= 0:
            if self.verbose:
                print(
                    f"[ utils/roa ] Skipping generation; already have {existing_runs} runs."
                )
            return self.final_states

        if existing_runs > 0:
            if self.verbose:
                print(
                    f"[ utils/roa ] Continuing from run {existing_runs + 1}/{self._expected_n_runs}"
                )
        if existing_runs > 0 and not discard_trajectories:
            warnings.warn(
                "Continuing from previous data, cannot retain previously generated trajectories. "
                "Stored trajectories will only contain newly generated data."
            )

        total_trajectories = self.start_states.shape[0] * runs_needed
        if total_trajectories > 1e4 and not discard_trajectories:
            warnings.warn(
                f"Generating and storing {total_trajectories} trajectories may consume a lot of memory."
            )

        final_state_runs, trajectory_runs = generator.generate_multiple_runs(
            start_states=self.start_states,
            n_runs=runs_needed,
            batch_size=self.batch_size,
            discard_trajectories=discard_trajectories,
            save=save,
            run_offset=existing_runs,
            metadata_n_runs=self._expected_n_runs,
            max_path_length=self.max_path_length,
            horizon_length=self.horizon_length,
            conditional_sample_kwargs=self.conditional_sample_kwargs,
            post_process_fns=self.inference_params["post_process_fns"],
            post_process_fn_kwargs=self.inference_params["post_process_fn_kwargs"],
            verbose=self.verbose,
        )

        generated_final_states = final_state_runs.transpose(1, 0, 2)

        if existing_runs > 0 and self.final_states is not None:
            self.final_states = np.concatenate(
                [self.final_states, generated_final_states], axis=1
            )
        else:
            self.final_states = generated_final_states

        if not discard_trajectories:
            if trajectory_runs is None:
                raise ValueError("Expected trajectories but none were returned.")
            self.trajectories = trajectory_runs.transpose(1, 0, 2, 3)
        elif existing_runs == 0:
            self.trajectories = None

        self.n_runs = self.final_states.shape[1]
        return self.final_states

    def load_final_states(self, timestamp: str = None, parallel=True) -> None:
        """
        Loads final states from files on disk.

        This is an alternative to `generate_trajectories()` or `set_final_states()`.

        Args:
            timestamp (str, optional): The timestamp of the run to load. If None,
                                     the latest run is loaded. Defaults to None.
            parallel (bool, optional): Whether to use multiprocessing to load files.
                                     Defaults to True.

        **Required attributes:**
            - `model_path`, `final_state_directory`: Used to find the data directories.

        **Set/Modified attributes:**
            - `final_states`
            - `n_runs`
            - `timestamp`
        """
        final_states_run_first, loaded_runs, resolved_timestamp = (
            self.trajectory_generator.load_saved_final_states(
                expected_runs=self._expected_n_runs,
                timestamp=timestamp,
                parallel=parallel,
            )
        )

        self.final_states = final_states_run_first.transpose(1, 0, 2)
        self.n_runs = loaded_runs
        self.timestamp = resolved_timestamp

        if self.n_runs != self._expected_n_runs:
            warnings.warn(
                f"Loaded {self.n_runs} runs instead of {self._expected_n_runs}"
            )

    def set_final_states(self, final_states: np.ndarray):
        """
        Directly sets the final states from an in-memory numpy array.

        This is an alternative to `generate_trajectories()` or `load_final_states()`,
        useful when final states are already available. After calling this,
        you can proceed with analysis methods like `compute_outcome_labels()`.

        Args:
            final_states (np.ndarray): Array of final states with shape
                                       (num_start_states, n_runs, state_dim).

        **Set/Modified attributes:**
            - `final_states`
            - `n_runs`
        """
        self.final_states = final_states
        self.n_runs = final_states.shape[1]

    def compute_outcome_labels(self) -> np.ndarray:
        """
        Outcome-based labeling using the provided system. Only used when system is set.
        """
        if self.system is None:
            raise ValueError("No system provided for outcome labeling.")
        if self.final_states is None:
            raise ValueError("No final states available.")

        n, n_runs, dim = self.final_states.shape
        reshaped_final_states = self.final_states.reshape(-1, dim)

        labels = []
        for fs in reshaped_final_states:
            outcome = self.system.evaluate_final_state(fs)
            labels.append(self.system.outcome_to_index(outcome))

        self.outcome_labels = np.array(labels, dtype=np.int32).reshape(n, n_runs)
        return self.outcome_labels

    def compute_outcome_probabilities(self):
        """
        Computes probabilities over system-defined outcomes.
        """
        if self.system is None:
            raise ValueError("No system provided for outcome probabilities.")
        if self.outcome_labels is None:
            self.compute_outcome_labels()

        n_points, n_runs = self.outcome_labels.shape
        outcomes = list(self.system.valid_outcomes)
        probs = np.zeros((n_points, len(outcomes)), dtype=np.float32)

        for idx, outcome in enumerate(outcomes):
            probs[:, idx] = (
                np.sum(self.outcome_labels == outcome.index, axis=1) / n_runs
            )

        self.outcome_probabilities = probs
        return self.outcome_probabilities

    def derive_labels_from_outcomes(self) -> np.ndarray:
        """
        Map outcome-based predictions to coarse labels used for RoA evaluation.

        Convention:
            - Positive class (inside RoA): Outcome.SUCCESS
            - Negative class: any non-invalid outcome that is not SUCCESS
            - Invalid class: high probability on invalid outcomes or uncertain prediction

        This populates `predicted_labels` in the same label space as
        `expected_labels` / `labels_array` (typically {0, 1, invalid_label}).
        """
        if self.system is None:
            raise ValueError("System must be provided to derive labels from outcomes.")

        if self.outcome_probabilities is None:
            self.compute_outcome_probabilities()
        if getattr(self, "predicted_outcomes", None) is None:
            # Do not save here; script can control persistence
            self.predict_outcomes(save=False)

        if self.labels_array is None:
            raise ValueError("Labels array not computed. Call _load_params first.")

        n_points = self.outcome_probabilities.shape[0]
        labels = np.full(n_points, self.invalid_label, dtype=np.int32)

        # Identify SUCCESS outcome index, if present
        outcomes = list(self.system.valid_outcomes)
        success_idx = None
        for idx, outcome in enumerate(outcomes):
            if outcome == Outcome.SUCCESS:
                success_idx = idx
                break

        # Invalid outcome indices already resolved from metadata / params
        invalid_out_idx = set(
            int(i)
            for i in (self.invalid_outcome_indices or np.array([], dtype=np.int64))
        )

        negative_label = self.labels_array[0]
        positive_label = self.labels_array[1]

        for i, out_idx in enumerate(self.predicted_outcomes):
            if out_idx == -1:
                # Uncertain prediction -> invalid
                continue
            if out_idx in invalid_out_idx:
                # Explicit invalid outcome -> invalid
                continue
            if success_idx is not None and out_idx == success_idx:
                labels[i] = positive_label
            else:
                labels[i] = negative_label

        self.predicted_labels = labels
        return labels

    def _resolve_invalid_outcome_indices(self, specs):
        """
        Map outcome names/values from metadata or inference params to indices
        in the outcome probability matrix.
        """
        if self.system is None or not specs:
            return np.array([], dtype=np.int64)

        outcomes = list(self.system.valid_outcomes)
        indices = []

        for spec in specs:
            try:
                if isinstance(spec, str):
                    target = Outcome[spec]
                else:
                    target = Outcome(spec)
            except Exception:
                continue

            for idx, outcome in enumerate(outcomes):
                if outcome == target:
                    indices.append(idx)
                    break

        if not indices:
            return np.array([], dtype=np.int64)

        return np.array(sorted(set(indices)), dtype=np.int64)

    def predict_outcomes(self, save=True):
        """
        Predicts outcomes for each start state based on outcome probabilities.
        """
        if self.system is None:
            raise ValueError("No system provided for outcome prediction.")
        if self.outcome_probabilities is None:
            self.compute_outcome_probabilities()

        probs = self.outcome_probabilities
        n_points, n_outcomes = probs.shape
        threshold = (
            self._outcome_prob_threshold
            if self._outcome_prob_threshold is not None
            else 0.0
        )

        invalid_val = -1
        predicted = np.full(n_points, invalid_val, dtype=np.int32)

        invalid_indices = getattr(self, "invalid_outcome_indices", None)
        if invalid_indices is None:
            invalid_indices = np.array([], dtype=np.int64)

        if invalid_indices.size > 0:
            invalid_prob = probs[:, invalid_indices].sum(axis=1)
        else:
            invalid_prob = np.zeros(n_points, dtype=probs.dtype)

        all_indices = np.arange(n_outcomes)
        valid_indices = np.setdiff1d(all_indices, invalid_indices, assume_unique=True)

        if valid_indices.size > 0:
            valid_probs = probs[:, valid_indices]
            best_valid_idx_local = np.argmax(valid_probs, axis=1)
            best_valid_prob = valid_probs[np.arange(n_points), best_valid_idx_local]
            best_valid_cols = valid_indices[best_valid_idx_local]
        else:
            best_valid_prob = np.zeros(n_points, dtype=probs.dtype)
            best_valid_cols = np.full(n_points, invalid_val, dtype=np.int32)

        invalid_high = invalid_prob >= threshold
        valid_high = best_valid_prob >= threshold

        assign_mask = valid_high & ~invalid_high
        predicted[assign_mask] = best_valid_cols[assign_mask]

        self.predicted_outcomes = predicted
        self.outcome_invalid_indices = np.where(invalid_high)[0]
        self.outcome_uncertain_indices = np.where(~invalid_high & ~valid_high)[0]

        if save:
            # reuse existing saver by mapping to labels file with outcome indices
            self._setup_results_path()
            file_path = path.join(self.results_path, f"predicted_outcomes.txt")
            with open(file_path, "w") as f:
                f.write("# ")
                for i in range(self.start_states.shape[1]):
                    f.write(f"start_{i} ")
                f.write("outcome\n")
                for idx in range(self.start_states.shape[0]):
                    for val in self.start_states[idx]:
                        f.write(f"{val} ")
                    f.write(f"{self.predicted_outcomes[idx]}\n")
            if self.verbose:
                print(f"[ utils/roa ] Saved predicted outcomes to {file_path}")

        return self.predicted_outcomes

    def load_predicted_labels(self, results_path: str):
        """
        Loads predicted classification labels from a file.

        Args:
            results_path (str): The path to the results directory containing the labels file.

        **Set/Modified attributes:**
            - `results_path`
            - `predicted_labels`
            - `separatrix_indices`
        """
        if self.verbose:
            print(f"[ utils/roa ] Loading predicted labels")

        self.results_path = results_path

        file_path = path.join(results_path, f"predicted_labels.txt")

        self.predicted_labels = []

        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("#"):
                    continue
                data = line.split()
                predicted_label = int(data[-1])
                self.predicted_labels.append(predicted_label)

        self.predicted_labels = np.array(self.predicted_labels, dtype=np.int32)
        self.separatrix_indices = np.where(self.predicted_labels == self.invalid_label)[
            0
        ]

        if self.verbose:
            print(f"[ utils/roa ] Loaded {len(self.predicted_labels)} predicted labels")

    def load_predicted_outcomes(self, results_path: str):
        """
        Loads predicted outcomes from a file.
        """
        if self.verbose:
            print(f"[ utils/roa ] Loading predicted outcomes")

        file_path = path.join(results_path, f"predicted_outcomes.txt")
        outcomes = []
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                data = line.split()
                outcomes.append(int(data[-1]))
        self.predicted_outcomes = np.array(outcomes, dtype=np.int32)
        if self.verbose:
            print(
                f"[ utils/roa ] Loaded {len(self.predicted_outcomes)} predicted outcomes"
            )

    def compute_classification_results(self, save=True):
        """
        Computes classification metrics by comparing predicted labels to expected labels.

        Calculates metrics like accuracy, precision, recall, TP, FP, TN, FN.

        **Required attributes:**
            - `predicted_labels`
            - `expected_labels`: Loaded via `load_ground_truth()`.
            - `labels_array`

        **Set/Modified attributes:**
            - `tp_mask`, `tn_mask`, `fp_mask`, `fn_mask`: Masks for plotting results.

        Returns:
            dict: A dictionary containing the classification results.
        """
        self._setup_results_path()

        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")

        if self.labels_array is None:
            raise ValueError("Labels array not computed. Call set_labels first.")

        # Create a mask to filter out invalid ground truth labels
        valid_gt_mask = self.expected_labels != self.labels_array[-1]

        # Apply the mask to get a clean set of labels for metric calculation
        expected_labels_valid = self.expected_labels[valid_gt_mask]
        predicted_labels_for_valid_gt = self.predicted_labels[valid_gt_mask]

        # Count how many times we predicted invalid for a valid ground truth
        n_invalid_predictions = np.sum(
            predicted_labels_for_valid_gt == self.labels_array[-1]
        )

        fp = np.sum(
            (predicted_labels_for_valid_gt == self.labels_array[1])
            & (expected_labels_valid == self.labels_array[0])
        )
        fn = np.sum(
            (predicted_labels_for_valid_gt == self.labels_array[0])
            & (expected_labels_valid == self.labels_array[1])
        )
        tp = np.sum(
            (predicted_labels_for_valid_gt == self.labels_array[1])
            & (expected_labels_valid == self.labels_array[1])
        )
        tn = np.sum(
            (predicted_labels_for_valid_gt == self.labels_array[0])
            & (expected_labels_valid == self.labels_array[0])
        )

        # Compute masks for true positives, true negatives, false positives, and false negatives
        self.fp_mask = (self.predicted_labels == self.labels_array[1]) & (
            self.expected_labels == self.labels_array[0]
        )
        self.fn_mask = (self.predicted_labels == self.labels_array[0]) & (
            self.expected_labels == self.labels_array[1]
        )
        self.tp_mask = (self.predicted_labels == self.labels_array[1]) & (
            self.expected_labels == self.labels_array[1]
        )
        self.tn_mask = (self.predicted_labels == self.labels_array[0]) & (
            self.expected_labels == self.labels_array[0]
        )

        tp_rate = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        tn_rate = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        invalid_rate = (
            int(n_invalid_predictions)
            / (int(n_invalid_predictions) + int(tp) + int(tn) + int(fp) + int(fn))
            if (int(n_invalid_predictions) + int(tp) + int(tn) + int(fp) + int(fn)) > 0
            else np.nan
        )

        accuracy = (
            (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else np.nan
        )

        results = {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "invalid_predictions": int(n_invalid_predictions),
            "tp_rate": float(tp_rate),
            "tn_rate": float(tn_rate),
            "fp_rate": float(fp_rate),
            "fn_rate": float(fn_rate),
            "invalid_rate": float(invalid_rate),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
        }

        if save:
            # Create results directory if it doesn't exist
            os.makedirs(self.results_path, exist_ok=True)

            # Save classification results to JSON file
            results_fpath = path.join(self.results_path, "classification_results.json")
            with open(results_fpath, "w") as f:
                json.dump(results, f, indent=4)

            if self.verbose:
                print(f"[ utils/roa ] Classification results saved to {results_fpath}")

        if self.verbose:
            print(
                f"\n[ utils/roa ] Classification results for {self.model_path} | {self.n_runs} runs:\n"
            )
            for key, value in results.items():
                if key == "confusion_matrix":
                    print(f"{key}:")
                    # Get the labels for better readability
                    labels = results["confusion_matrix_labels"]

                    # Print header row with predicted labels
                    header = "True\\Pred |"
                    for label in labels:
                        header += f" {label:^7} |"
                    print(header)
                    print("-" * len(header))

                    # Print each row with the true label and confusion matrix values
                    for i, true_label in enumerate(labels):
                        row = f" {true_label:^7} |"
                        for j in range(len(labels)):
                            row += f" {value[i][j]:^7} |"
                        print(row)
                    print("-" * len(header))
                elif key == "class_metrics":
                    print(f"{key}:")
                    for class_label, metrics in value.items():
                        print(f"  {class_label}:")
                        for metric_name, metric_value in metrics.items():
                            print(f"    {metric_name}: {metric_value}")
                else:
                    print(f"{key}: {value}")

        return results


# Backwards-compatible alias for legacy code paths.
ROAEstimator = Classifier

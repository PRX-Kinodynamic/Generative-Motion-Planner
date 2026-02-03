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
    get_data_trajectories_path,
    get_method_name,
    load_inference_params,
    load_roa_labels,
    get_trajectory_attractor_labels,
    compute_actual_length,
)
from genMoPlan.utils.trajectory_generator import TrajectoryGenerator

class ROAEstimator:
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
    attractor_labels: np.ndarray = None
    label_probabilities: np.ndarray = None
    invalid_label: int = None
    attractors: dict = None
    _attractor_dist_threshold: float = None
    _attractor_prob_threshold: float = None
    labels_set: set = None
    labels_array: np.ndarray = None
    tp_mask: np.ndarray = None
    tn_mask: np.ndarray = None
    fp_mask: np.ndarray = None
    fn_mask: np.ndarray = None
    
    def __init__(self, 
        dataset:str, 
        model_state_name: str = "best.pt", 
        model_path: str = None, 
        n_runs: int = None, 
        batch_size: int = None, 
        verbose: bool = True, 
        num_batches: int = None,
        device: str ='cuda',
    ):
        self.dataset = dataset
        self.model_path = model_path
        self.verbose = verbose
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device

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
        self.n_runs = self._orig_n_runs if self._orig_n_runs is not None else self.inference_params["n_runs"]
        self._expected_n_runs = self.n_runs
        self.batch_size = self.batch_size if self.batch_size is not None else self.inference_params["batch_size"]

        self._attractor_dist_threshold = self.inference_params["attractor_dist_threshold"]
        self._attractor_prob_threshold = self.inference_params["attractor_prob_threshold"]
        
        self.invalid_label = self.inference_params["invalid_label"]
        self.max_path_length = self.inference_params["max_path_length"]
        self.final_state_directory = self.inference_params["final_state_directory"]

        if self.trajectory_generator is not None:
            self.conditional_sample_kwargs = dict(self.trajectory_generator.conditional_sample_kwargs or {})
        else:
            self.conditional_sample_kwargs = (
                self.inference_params[self.method_name]
                if self.method_name in self.inference_params
                else {}
            )
        self.ground_truth_filter_fn = self.inference_params["ground_truth_filter_fn"] if "ground_truth_filter_fn" in self.inference_params else None
        self.attractors = self.inference_params["attractors"] if "attractors" in self.inference_params else None
        self.attractor_classification_fn = self.inference_params["attractor_classification_fn"] if "attractor_classification_fn" in self.inference_params else get_trajectory_attractor_labels

        self.labels_set = set(list(self.attractors.values())) if self.attractors is not None else set(self.inference_params["attractor_labels"])
        self.labels_array = np.array([*self.labels_set, self.invalid_label])    
        
        assert self.attractors is not None or self.attractor_classification_fn is not None, "Either attractors or attractor classification function must be provided"

    @property
    def trajectory_generator(self) -> TrajectoryGenerator:
        if self._trajectory_generator is None:
            raise ValueError("Trajectory generation requires `model_path` to be provided.")
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
            self.results_path = path.join(self.model_path, "results", self.inference_params["results_name"](self.inference_params, self.method_name))
        else:
            self.results_path = path.join(self.model_path, "results", self.inference_params["results_name"])

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self._save_inference_params()

    def _setup_traj_plot_path(self):
        if self.traj_plot_path is not None:
            return
        
        if callable(self.inference_params["results_name"]):
            self.traj_plot_path = path.join(self.model_path, "viz_trajs", self.inference_params["results_name"](self.inference_params, self.method_name))
        else:
            self.traj_plot_path = path.join(self.model_path, "viz_trajs", self.inference_params["results_name"])

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

        file_path = path.join(self.results_path, f'predicted_attractor_labels.txt')

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

    def set_attractor_dist_threshold(self, attractor_dist_threshold: float):
        self._attractor_dist_threshold = attractor_dist_threshold
        self.inference_params["attractor_dist_threshold"] = attractor_dist_threshold

    def set_attractor_prob_threshold(self, attractor_prob_threshold: float):
        self._attractor_prob_threshold = attractor_prob_threshold
        self.inference_params["attractor_prob_threshold"] = attractor_prob_threshold

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        self.inference_params["batch_size"] = batch_size
        if self.trajectory_generator is not None:
            self.trajectory_generator.set_batch_size(batch_size)

    def set_horizon_and_max_path_lengths(self, horizon_length: Optional[int] = None, *, max_path_length: Optional[int] = None, num_inference_steps: Optional[int] = None):
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
            actual_hist = compute_actual_length(self.history_length, getattr(self, "stride", 1))
            actual_horz = compute_actual_length(self.horizon_length, getattr(self, "stride", 1))
            self.max_path_length = actual_hist + (num_inference_steps * actual_horz)

        self.inference_params["max_path_length"] = self.max_path_length

    def reset(self, for_analysis: bool = False):
        """
        Resets the ROAEstimator for new runs while retaining the ground truth data

        This method is useful when you want to perform a new analysis with different parameters
        on the same ground truth data without re-initializing the entire object.

        Args:
            for_analysis: If True, only the analysis results will be reset but the run data
                          (e.g., final_states, trajectories) will be retained.

        **Set/Modified attributes:**
            - `results_path`
            - `traj_plot_path`
            - `attractor_labels`
            - `predicted_labels`
            - `uncertain_indices`
            - `separatrix_indices`
            - `label_probabilities`
            - `tp_mask`, `tn_mask`, `fp_mask`, `fn_mask`
            - (if `for_analysis` is False): `final_states`, `trajectories`, `n_runs`
        """
        self.results_path = None
        self.traj_plot_path = None
        self._load_params()

        self.attractor_labels = None
        self.predicted_labels = None
        self.uncertain_indices = None
        self.separatrix_indices = None
        self.label_probabilities = None
        self.tp_mask = None
        self.tn_mask = None
        self.fp_mask = None
        self.fn_mask = None

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
        
    def plot_attractor_probabilities(self, s=0.1):
        """
        Plots the computed attractor probabilities as scatter plots.

        **Required attributes:**
            - `label_probabilities`
            - `start_states`
            - `labels_set`
        """
        self._setup_results_path()

        if self.verbose:
            print(f"[ utils/roa ] Plotting attractor probabilities")
        
        if self.label_probabilities is None:
            raise ValueError("No attractor probabilities available")
        
        if self.start_states.shape[1] > 2:
            raise ValueError("Cannot plot probabilities for more than 2D start points")
        
        os.makedirs(self.results_path, exist_ok=True)
        
        for i, label in enumerate(self.labels_set):
            plt.figure(figsize=(10, 8))
            plt.scatter(self.start_states[:, 0], self.start_states[:, 1], 
                    c=self.label_probabilities[:, i], s=s, cmap='RdBu')
            plt.colorbar()
            plt.title(f'Probability of label {label}')
            plt.savefig(path.join(self.results_path, f"probability_{label}.png"))
            plt.close()

        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_states[:, 0], self.start_states[:, 1], 
                c=self.label_probabilities[:, -1], s=s, cmap='RdBu')
        plt.colorbar()
        plt.title('Probability of invalid label')
        plt.savefig(path.join(self.results_path, f"probability_invalid.png"))
        plt.close()
    
    def plot_predicted_attractor_labels(self, s=0.1):
        """
        Plots the predicted attractor labels.

        **Required attributes:**
            - `predicted_labels`
            - `start_states`
            - `labels_array`, `labels_set`
            - `uncertain_indices` (optional)
        """
        self._setup_results_path()

        if self.verbose:
            print(f"[ utils/roa ] Plotting predicted attractor labels")

        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")
        
        if self.start_states.shape[1] > 2:
            raise ValueError("Cannot plot probabilities for more than 2D start points")
        
        invalid_label = self.labels_array[-1]
        
        os.makedirs(self.results_path, exist_ok=True)
        
        for label in self.labels_set:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.start_states[:, 0], self.start_states[:, 1], 
                       c=self.predicted_labels == label, s=s, cmap='viridis')
            plt.title(f'{label} Label Predictions')
            plt.savefig(path.join(self.results_path, f"predicted_{label}.png"))
            plt.close()

        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_states[:, 0], self.start_states[:, 1], 
                   c=self.predicted_labels == invalid_label, s=s, cmap='viridis')
        plt.title('Invalid Label Predictions')
        plt.savefig(path.join(self.results_path, f"predicted_invalid_label.png"))
        plt.close()

        if self.uncertain_indices is not None:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.start_states[self.uncertain_indices, 0], self.start_states[self.uncertain_indices, 1], 
                       c='red', s=s)
            plt.title('Uncertain points')
            plt.savefig(path.join(self.results_path, f"uncertain_points.png"))
            plt.close()

        else:
            print(f"[ utils/roa ] No uncertain points found or loaded")

        if self.verbose:    
            print(f"[ utils/roa ] Attractor probabilities plotted in {self.results_path}")

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
        
        plt.scatter(self.start_states[success_mask, 0], self.start_states[success_mask, 1], 
                   c=labels_to_plot[success_mask], s=s, cmap=self.SUCCESS_COLOR_MAP)
        plt.scatter(self.start_states[failure_mask, 0], self.start_states[failure_mask, 1], 
                   c=labels_to_plot[failure_mask], s=s, cmap=self.FAILURE_COLOR_MAP)
        plt.scatter(self.start_states[separatrix_mask, 0], self.start_states[separatrix_mask, 1], 
                   c=labels_to_plot[separatrix_mask], s=s, cmap=self.SEPARATRIX_COLOR_MAP)

        plt.title('RoAs')
        plt.savefig(path.join(self.results_path, f"roas.png"))
        plt.close()

        if plot_separatrix:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.start_states[self.separatrix_indices, 0], self.start_states[self.separatrix_indices, 1], 
                       c='red', s=s)
            plt.title('Separatrix')
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

        if self.tp_mask is None or self.tn_mask is None or self.fp_mask is None or self.fn_mask is None:
            raise ValueError("Classification results not computed. Call compute_prediction_metrics first.")
        
        if self.start_states.shape[1] > 2:
            raise ValueError("Cannot plot classification results for more than 2D start points")

        # Plot true positives
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_states[self.tp_mask, 0], self.start_states[self.tp_mask, 1], 
                   c=self.predicted_labels[self.tp_mask], s=s, cmap=self.SUCCESS_COLOR_MAP)
        plt.title('True Positives')
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "true_positives.png"))
        plt.close()
        
        # Plot true negatives
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_states[self.tn_mask, 0], self.start_states[self.tn_mask, 1], 
                   c=self.predicted_labels[self.tn_mask], s=s, cmap=self.FAILURE_COLOR_MAP)
        plt.title('True Negatives')
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "true_negatives.png"))
        plt.close()
        
        # Plot false positives
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_states[self.fp_mask, 0], self.start_states[self.fp_mask, 1], 
                   c=self.predicted_labels[self.fp_mask], s=s, cmap=self.FP_COLOR_MAP)
        plt.title('False Positives')
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "false_positives.png"))
        plt.close()
        
        # Plot false negatives
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_states[self.fn_mask, 0], self.start_states[self.fn_mask, 1], 
                   c=self.predicted_labels[self.fn_mask], s=s, cmap=self.FN_COLOR_MAP)
        plt.title('False Negatives')
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "false_negatives.png"))
        plt.close()

        # Combined plots with all incorrect classifications
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_states[self.fp_mask, 0], self.start_states[self.fp_mask, 1], 
                   c=self.predicted_labels[self.fp_mask], s=s, cmap=self.FP_COLOR_MAP, label='False Positives')
        plt.scatter(self.start_states[self.fn_mask, 0], self.start_states[self.fn_mask, 1], 
                   c=self.predicted_labels[self.fn_mask], s=s, cmap=self.FN_COLOR_MAP, label='False Negatives')
        plt.title('Incorrect Classifications')
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "incorrect_classifications.png"))
        plt.close()
        
        # Combined plot with all classifications
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_states[self.tp_mask, 0], self.start_states[self.tp_mask, 1], 
                   c=self.predicted_labels[self.tp_mask], s=s, cmap=self.SUCCESS_COLOR_MAP, label='True Positives')
        plt.scatter(self.start_states[self.tn_mask, 0], self.start_states[self.tn_mask, 1], 
                   c=self.predicted_labels[self.tn_mask], s=s, cmap=self.FAILURE_COLOR_MAP, label='True Negatives')
        plt.scatter(self.start_states[self.fp_mask, 0], self.start_states[self.fp_mask, 1], 
                   c=self.predicted_labels[self.fp_mask], s=s, cmap=self.FP_COLOR_MAP, label='False Positives')
        plt.scatter(self.start_states[self.fn_mask, 0], self.start_states[self.fn_mask, 1], 
                   c=self.predicted_labels[self.fn_mask], s=s, cmap=self.FN_COLOR_MAP, label='False Negatives')
        plt.scatter(self.start_states[self.separatrix_indices, 0], self.start_states[self.separatrix_indices, 1], 
                   c=self.predicted_labels[self.separatrix_indices], s=s, cmap=self.SEPARATRIX_COLOR_MAP, label='Separatrix')
        plt.title('Classification Results')
        plt.xlim(self.start_states[:, 0].min(), self.start_states[:, 0].max())
        plt.ylim(self.start_states[:, 1].min(), self.start_states[:, 1].max())
        plt.savefig(path.join(self.results_path, "classification_results.png"))
        plt.close()

        if self.verbose:
            print(f"[ utils/roa ] Classification result plots saved in {self.results_path}")

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
            print('[ scripts/estimate_roa ] Loading ground truth')
        
        start_states, expected_labels = load_roa_labels(self.dataset)

        if self.ground_truth_filter_fn is not None and self.model_args is not None:
            start_states, expected_labels = self.ground_truth_filter_fn(start_states, expected_labels, self.model_args)

        self.start_states = start_states
        self.expected_labels = expected_labels

        if self.num_batches is not None:
            self.set_batch_size(self.start_states.shape[0] // self.num_batches)

        if self.verbose:
            print(f"[ utils/roa ] Loaded {self.start_states.shape[0]} ground truth data points")

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
            raise ValueError("Start states must be loaded before generating trajectories.")

        if self.n_runs is None:
            raise ValueError("Number of runs (`n_runs`) must be defined before generation.")

        existing_runs = 0 if self.final_states is None else self.final_states.shape[1]
        runs_needed = self._expected_n_runs - existing_runs

        if runs_needed <= 0:
            if self.verbose:
                print(f"[ utils/roa ] Skipping generation; already have {existing_runs} runs.")
            return self.final_states

        if existing_runs > 0:
            if self.verbose:
                print(f"[ utils/roa ] Continuing from run {existing_runs + 1}/{self._expected_n_runs}")
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
            self.final_states = np.concatenate([self.final_states, generated_final_states], axis=1)
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
        final_states_run_first, loaded_runs, resolved_timestamp = self.trajectory_generator.load_saved_final_states(
            expected_runs=self._expected_n_runs,
            timestamp=timestamp,
            parallel=parallel,
        )

        self.final_states = final_states_run_first.transpose(1, 0, 2)
        self.n_runs = loaded_runs
        self.timestamp = resolved_timestamp

        if self.n_runs != self._expected_n_runs:
            warnings.warn(f"Loaded {self.n_runs} runs instead of {self._expected_n_runs}")
         
    def set_final_states(self, final_states: np.ndarray):
        """
        Directly sets the final states from an in-memory numpy array.

        This is an alternative to `generate_trajectories()` or `load_final_states()`,
        useful when final states are already available. After calling this,
        you can proceed with analysis methods like `compute_attractor_labels()`.

        Args:
            final_states (np.ndarray): Array of final states with shape
                                       (num_start_states, n_runs, state_dim).

        **Set/Modified attributes:**
            - `final_states`
            - `n_runs`
        """
        self.final_states = final_states
        self.n_runs = final_states.shape[1]
   
    def compute_attractor_labels(self) -> np.ndarray:
        """
        Computes attractor labels for each final state of each run in `final_states`.

        This is useful if you have final states but not the corresponding labels, e.g.,
        after using `set_final_states()` or `load_final_states()`.

        **Required attributes:**
            - `final_states`: The final states of trajectories.
            - `attractors`, `attractor_dist_threshold`, `invalid_label`: From inference params.

        **Set/Modified attributes:**
            - `attractor_labels`: Computed labels for each final state.

        Returns:
            np.ndarray: The computed attractor labels.
        """
        

        if self.verbose:
            print(f"[ utils/roa ] Computing attractor labels")
        if self.final_states is None:
            raise ValueError("No final states available.")
        
        n, n_runs, dim = self.final_states.shape
        reshaped_final_states = self.final_states.reshape(-1, dim)
        
        reshaped_labels = self.attractor_classification_fn(
            reshaped_final_states, self.attractors, self._attractor_dist_threshold, self.invalid_label, verbose=self.verbose
        )
        
        self.attractor_labels = reshaped_labels.reshape(n, n_runs)
        self.attractor_labels = np.array(self.attractor_labels, dtype=np.int32)
        
        return self.attractor_labels

    def compute_attractor_probabilities(self):
        """
        Computes the probability of each start state converging to each attractor.

        This method calculates, for each start state, the frequency with which it
        ended up at each attractor over `n_runs`.

        **Required attributes:**
            - `attractor_labels`: The label of the attractor for each run.
            - `start_states`: Needed for plotting.
            - `labels_set`, `labels_array`: Information about possible labels.
            - `n_runs`: The number of runs to compute probabilities over.

        **Set/Modified attributes:**
            - `label_probabilities`: An array of shape (n_points, n_labels) with probabilities.

        Returns:
            np.ndarray: The computed label probabilities.
        """
        if self.verbose:
            print(f"[ utils/roa ] Computing attractor probabilities")
        if self.attractor_labels is None:
            raise ValueError("No attractor labels available")
        
        if self.labels_set is None or self.labels_array is None:
            raise ValueError("Labels set not computed. Run set_labels first.")
            
        n_points = len(self.start_states)
        
        self.label_probabilities = np.zeros((n_points, len(self.labels_set) + 1))
        
        invalid_label = self.labels_array[-1]
        
        # Compute probability of invalid label
        self.label_probabilities[:, -1] = np.sum(self.attractor_labels == invalid_label, axis=1) / self.n_runs

        # Compute probability of each attractor label
        for i, label in enumerate(self.labels_set):
            self.label_probabilities[:, i] = np.sum(self.attractor_labels == label, axis=1) / self.n_runs

        # Check that probabilities sum to 1 (allowing for floating point error)
        assert np.allclose(np.sum(self.label_probabilities, axis=1), 1.0, rtol=1e-5, atol=1e-6), "Probabilities do not sum to 1"
        
        return self.label_probabilities
    
    def predict_attractor_labels(self, save=True):
        """
        Predicts the attractor for each start state based on label probabilities.

        A label is assigned if its probability is greater than or equal to `attractor_prob_threshold`.
        If no label meets the threshold, it is marked as uncertain/separatrix.

        **Required attributes:**
            - `label_probabilities`: If not present, `compute_attractor_probabilities` is called.
            - `labels_array`: Information about possible labels.
            - `start_states`: For determining the number of points.
            - `attractor_prob_threshold`: The probability threshold for assigning a label.

        **Set/Modified attributes:**
            - `predicted_labels`: The final predicted label for each start state.
            - `uncertain_indices`: Indices of points where no prediction could be made.
            - `separatrix_indices`: Indices of points predicted as the invalid label.

        Returns:
            np.ndarray: The array of predicted labels.
        """
        if self.verbose:
            print(f"[ utils/roa ] Predicting attractor labels")
        invalid_label = self.labels_array[-1]

        if self.label_probabilities is None:
            self.compute_attractor_probabilities()
        
        if self.labels_array is None:
            raise ValueError("Labels array not computed. Call compute_attractor_labels first.")
        
        predicted_labels = np.full(len(self.start_states), invalid_label, dtype=np.int32)
        
        # Assign label with highest probability exceeding threshold
        for i in range(len(self.labels_array[:-1])):
            mask = self.label_probabilities[:, i] >= self._attractor_prob_threshold
            predicted_labels[mask] = self.labels_array[i]

        self.predicted_labels = predicted_labels

        # Store indices of uncertain points - points with no probability exceeding threshold
        uncertain_indices = np.where(np.sum(self.label_probabilities >= self._attractor_prob_threshold, axis=1) == 0)[0]
        self.uncertain_indices = uncertain_indices

        self.separatrix_indices = np.where(predicted_labels == invalid_label)[0]
        
        if self.verbose:
            print(f"[ utils/roa ] Found {len(uncertain_indices)} uncertain points")
            print(f"[ utils/roa ] Total separatrix points: {len(self.separatrix_indices)}")

        if save:
            self._save_predicted_labels()
        
        return predicted_labels
   
    def load_predicted_labels(self, results_path: str):
        """
        Loads predicted attractor labels from a file.
        Alternative to `predict_attractor_labels()`.

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

        file_path = path.join(results_path, f'predicted_attractor_labels.txt')

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
        self.separatrix_indices = np.where(self.predicted_labels == self.invalid_label)[0]
        
        if self.verbose:
            print(f"[ utils/roa ] Loaded {len(self.predicted_labels)} predicted labels")
 
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
        n_invalid_predictions = np.sum(predicted_labels_for_valid_gt == self.labels_array[-1])
        
        fp = np.sum((predicted_labels_for_valid_gt == self.labels_array[1]) & (expected_labels_valid == self.labels_array[0]))
        fn = np.sum((predicted_labels_for_valid_gt == self.labels_array[0]) & (expected_labels_valid == self.labels_array[1]))
        tp = np.sum((predicted_labels_for_valid_gt == self.labels_array[1]) & (expected_labels_valid == self.labels_array[1]))
        tn = np.sum((predicted_labels_for_valid_gt == self.labels_array[0]) & (expected_labels_valid == self.labels_array[0]))

        # Compute masks for true positives, true negatives, false positives, and false negatives
        self.fp_mask = (self.predicted_labels == self.labels_array[1]) & (self.expected_labels == self.labels_array[0])
        self.fn_mask = (self.predicted_labels == self.labels_array[0]) & (self.expected_labels == self.labels_array[1])
        self.tp_mask = (self.predicted_labels == self.labels_array[1]) & (self.expected_labels == self.labels_array[1])
        self.tn_mask = (self.predicted_labels == self.labels_array[0]) & (self.expected_labels == self.labels_array[0])

        tp_rate = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        tn_rate = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        invalid_rate = int(n_invalid_predictions) / (int(n_invalid_predictions) + int(tp) + int(tn) + int(fp) + int(fn)) if (int(n_invalid_predictions) + int(tp) + int(tn) + int(fp) + int(fn)) > 0 else np.nan

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

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
            "f1_score": float(f1_score)
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
            print(f"\n[ utils/roa ] Classification results for {self.model_path} | {self.n_runs} runs:\n")
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

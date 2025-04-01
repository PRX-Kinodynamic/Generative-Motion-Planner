import gc
import glob
import multiprocessing
import os
from os import path
import json
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt

from tqdm import tqdm

def _load_attractor_labels(file_path):
    labels = []
    final_states = []
    start_points = []
    start_dim = 0
    final_dim = 0

    try:
        with open(file_path, "r") as f:
            for line in f:
                line_data = line.strip().split(' ')
                if line[0] == "#" or line[0] == "s":
                    for i in range(len(line_data)):
                        if line_data[i].startswith("start"):
                            start_dim += 1
                        elif line_data[i].startswith("final"):
                            final_dim += 1
                    
                    continue

                start_points.append([np.float32(value) for value in line_data[0:start_dim]])
                labels.append(int(float(line_data[start_dim])))
                final_states.append([np.float32(value) for value in line_data[start_dim + 1:start_dim + final_dim + 1]])

        return np.array(labels, dtype=np.int32), np.array(final_states, dtype=np.float32)
    except Exception as e:
        print(f"[ utils/roa ] Error loading attractor labels from {file_path}: {e}")
        return None, None

def _get_latest_timestamp(exp_path):
    if not os.path.exists(exp_path) or len(os.listdir(exp_path)) == 0:
        return None
    
    timestamps = [path.basename(d) for d in glob.glob(path.join(exp_path, "generated_trajectories", "*"))]
    return max(timestamps)

def _load_ground_truth(dataset):
    print('[ scripts/estimate_roa ] Loading ground truth')
    roa_labels_fpath = path.join("data_trajectories", dataset, "roa_labels.txt")

    data = []
    
    if os.path.exists(roa_labels_fpath):
        with open(roa_labels_fpath, "r") as f:
            for line in f:
                line_data = line.strip().split(' ')[1:]
                data.append([
                    np.float32(line_data[0]),
                    np.float32(line_data[1]),
                    int(line_data[2])
                ])
    else:
        raise FileNotFoundError(f"File {roa_labels_fpath} not found")
    
    return np.array(data, dtype=np.float32)


class ROAEstimator:
    model_state_name: str = "best.pt"
    exp_path: str = None
    gen_traj_path: str = None
    results_path: str = None
    method_name: str = None
    _timestamp: str = None
    
    dataset: str = None
    model: torch.nn.Module = None
    model_args: dict = None
    batch_size: int = None
    max_path_length: int = None
    conditional_sample_kwargs: dict = None


    start_points: np.ndarray = None
    expected_labels: np.ndarray = None
    n_runs: int = None
    _expected_n_runs: int = None
    trajectories: np.ndarray = None
    final_states: np.ndarray = None
    predicted_labels: np.ndarray = None
    uncertain_indices: np.ndarray = None

    # Classification
    roa_estimation_params: dict = None
    attractor_labels: np.ndarray = None
    label_probabilities: np.ndarray = None
    invalid_label: int = None
    attractors: dict = None
    attractor_dist_threshold: float = None
    attractor_prob_threshold: float = None
    labels_set: set = None
    labels_array: np.ndarray = None
    
    def __init__(self, dataset:str, model_state_name: str = "best.pt", exp_path: str = None, n_runs: int = None, batch_size: int = None):
        self.dataset = dataset
        self.exp_path = exp_path
        
        self._load_model(model_state_name)
        self._load_params(n_runs, batch_size)

    def init_ground_truth(self):
        ground_truth = _load_ground_truth(self.dataset)
        self.start_points = ground_truth[:, :2]
        self.expected_labels = ground_truth[:, 2]

            
    def _load_model(self, model_state_name: str = "best.pt"):
        from genMoPlan.utils import load_model

        self.model, self.model_args = load_model(self.exp_path, model_state_name, verbose=True)
    
    def _load_params(self, n_runs: int, batch_size: int):
        from genMoPlan.utils import load_roa_estimation_params, get_method_name

        self.roa_estimation_params = load_roa_estimation_params(self.dataset)

        self.n_runs = n_runs if n_runs is not None else self.roa_estimation_params["n_runs"]
        self._expected_n_runs = self.n_runs
        self.batch_size = batch_size if batch_size is not None else self.roa_estimation_params["batch_size"]

        self.attractor_dist_threshold = self.roa_estimation_params["attractor_dist_threshold"]
        self.attractor_prob_threshold = self.roa_estimation_params["attractor_prob_threshold"]
        self.attractors = self.roa_estimation_params["attractors"]
        self.invalid_label = self.roa_estimation_params["invalid_label"]
        self.max_path_length = self.roa_estimation_params["max_path_length"]

        self.labels_set = set(list(self.attractors.values()))
        self.labels_array = np.array([*self.labels_set, self.invalid_label])    

        self.method_name = get_method_name(self.model_args)
        self.conditional_sample_kwargs = self.roa_estimation_params[self.method_name] if self.method_name in self.roa_estimation_params else {}

    def _setup_results_path(self):
        from genMoPlan.utils import generate_timestamp

        if self.results_path is not None:
            return

        self.results_path = path.join(self.exp_path, "results", generate_timestamp())

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self._save_roa_estimation_params()

    def _save_roa_estimation_params(self, verbose=True):
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k): convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            else:
                return obj
        
        json_safe_params = convert_keys(self.roa_estimation_params)

        json_safe_params["n_runs"] = self.n_runs
        json_safe_params["batch_size"] = self.batch_size

        json_fpath = path.join(self.results_path, "roa_estimation_params.json")
        
        with open(json_fpath, "w") as f:
            json.dump(json_safe_params, f, indent=4)

        if verbose:
            print(f"[ utils/roa ] ROA estimation params saved in {json_fpath}")
    
    @property
    def timestamp(self):
        return self._timestamp
    
    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value
        self.gen_traj_path = path.join(self.exp_path, "generated_trajectories", self._timestamp)

        if not os.path.exists(self.gen_traj_path):
            os.makedirs(self.gen_traj_path)
        
    
    def _generate_single_run_trajs(self, run_idx, compute_labels, verbose, discard_trajectories, save):
        from genMoPlan.utils import generate_trajectories

        results = generate_trajectories(
            self.model, self.model_args, self.start_points, 
            max_path_length=self.max_path_length,
            only_execute_next_step=False, verbose=verbose, batch_size=self.batch_size, 
            only_return_final_states=discard_trajectories,
            conditional_sample_kwargs=self.conditional_sample_kwargs
        )
        
        if discard_trajectories:
            final_states = results
            generated_trajs = None
        else:
            final_states = results[:, -1].copy()
            generated_trajs = results

        assert final_states.shape == self.start_points.shape

        if compute_labels:
            from genMoPlan.utils import get_trajectory_attractor_labels

            attractor_labels = get_trajectory_attractor_labels(final_states, self.attractors, self.attractor_dist_threshold, self.invalid_label)
            self.attractor_labels.append(attractor_labels)

            if save:
                self._save_single_run_data(run_idx, final_states, attractor_labels)
                
        if not discard_trajectories:
            self.trajectories.append(generated_trajs)
        else:
            del generated_trajs

        gc.collect()
        
        try:
            params = list(self.model.parameters())
            if params and params[0].device.type == "cuda" and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            warnings.warn(f"Error clearing CUDA cache: {e}")

        self.final_states.append(final_states)
    
    def _save_single_run_data(self, run_idx, final_states, attractor_labels):
        file_path = path.join(self.gen_traj_path, f"attractor_labels_{run_idx}.txt")
        with open(file_path, "w") as f:
            f.write("# ")
            for i in range(self.start_points.shape[1]):
                f.write(f"start_{i} ")
            f.write("label ")
            for i in range(final_states.shape[1]):
                f.write(f"final_{i} ")
            f.write("\n")
            
            for start_point_idx in range(self.start_points.shape[0]):
                for i in range(self.start_points.shape[1]):
                    f.write(f"{self.start_points[start_point_idx, i]} ")
                    
                f.write(f"{attractor_labels[start_point_idx]} ")
                
                for i in range(final_states.shape[1]):
                    f.write(f"{final_states[start_point_idx, i]} ")
                
                f.write("\n")

        print(f"[ utils/roa ] Attractor labels saved in {file_path}")
    
    def generate_trajectories(
            self,
            compute_labels: bool = True,
            discard_trajectories: bool = True, 
            verbose: bool = False,
            save: bool = False,
        ):
        from genMoPlan.utils.progress import ETAIterator

        if save and not compute_labels:
            warnings.warn("Cannot save final states without computing attractor labels")
            compute_labels = True

        if self._timestamp is None:
            from genMoPlan.utils import generate_timestamp

            self.timestamp = generate_timestamp()

        start_idx = 0
        merge_prev_data = False
        prev_final_states = None
        prev_attractor_labels = None
        n_runs_to_generate = self.n_runs

        if self.final_states is not None and self.attractor_labels is not None:
            start_idx = self.n_runs
            merge_prev_data = True
            prev_final_states = self.final_states.copy()
            prev_attractor_labels = self.attractor_labels.copy()
            n_runs_to_generate = self._expected_n_runs - self.n_runs
            print(f"[ utils/roa ] Continuing from run {start_idx + 1}/{start_idx + self.n_runs}")

            if not discard_trajectories:
                warnings.warn("Continuing from previous data, cannot load previous trajectories. Stored trajectories will only contain new data.")

        total_trajectories = self.start_points.shape[0] * n_runs_to_generate
        
        if total_trajectories > 1e4 and not discard_trajectories:
            warnings.warn(f"Generating and storing {total_trajectories} trajectories may consume a lot of memory.")
        
        if not discard_trajectories:
            self.trajectories = []

        self.final_states = []
        self.attractor_labels = []

        run_range = range(start_idx, start_idx + n_runs_to_generate)
        eta_iter = ETAIterator(iter(run_range), n_runs_to_generate)
        
        for run_idx in eta_iter:
            print(f"[ utils/roa ] Run {run_idx+1}/{start_idx + n_runs_to_generate} (Remaining Time: {eta_iter.eta_formatted})")
            
            self._generate_single_run_trajs(
                run_idx, compute_labels, verbose, discard_trajectories, save
            )

        if not discard_trajectories:
            self.trajectories = np.array(self.trajectories, dtype=np.float32) # (n_runs, num_start_states, path_length, dim)
            self.trajectories = np.transpose(self.trajectories, (1, 0, 2, 3)) # (num_start_states, n_runs, path_length, dim)
        
        self.final_states = np.array(self.final_states, dtype=np.float32) # (n_runs, num_start_states, dim)
        self.final_states = np.transpose(self.final_states, (1, 0, 2)) # (num_start_states, n_runs, dim)

        if merge_prev_data:
            self.final_states = np.concatenate([prev_final_states, self.final_states], axis=1)
            self.n_runs = self._expected_n_runs

        if compute_labels:
            self.attractor_labels = np.array(self.attractor_labels, dtype=np.int32) # (n_runs, num_start_states)
            self.attractor_labels = np.transpose(self.attractor_labels, (1, 0)) # (num_start_states, n_runs)

            if merge_prev_data:
                self.attractor_labels = np.concatenate([prev_attractor_labels, self.attractor_labels], axis=1)

        assert self.final_states.shape[1] == self.n_runs

        return self.final_states
    
    def compute_attractor_labels(self, verbose=True):
        from genMoPlan.utils import get_trajectory_attractor_labels

        if verbose:
            print(f"[ utils/roa ] Computing attractor labels")
        if self.final_states is None:
            raise ValueError("No final states available.")
        
        n, n_runs, dim = self.final_states.shape
        reshaped_final_states = self.final_states.reshape(-1, dim)
        
        reshaped_labels = get_trajectory_attractor_labels(
            reshaped_final_states, self.attractors, self.attractor_dist_threshold, self.invalid_label
        )
        
        self.attractor_labels = reshaped_labels.reshape(n, n_runs)
        self.attractor_labels = np.array(self.attractor_labels, dtype=np.int32)
        
        return self.attractor_labels
    
    def save_final_states(self, verbose=True):
        if verbose:
            print(f"[ utils/roa ] Saving final states")
        if self.final_states is None:
            raise ValueError("No final states available")
        if self.attractor_labels is None:
            raise ValueError("No attractor labels available")
        
        for run_idx in range(self.final_states.shape[1]):
            final_states = self.final_states[:, run_idx, :]
            attractor_labels = self.attractor_labels[:, run_idx]

            self._save_single_run_data(run_idx, final_states, attractor_labels)
        
        print(f"[ utils/roa ] Attractor labels saved in {self.gen_traj_path}")
    
    def load_final_states(self, timestamp: str = None, parallel=True):
        if timestamp is None:
            self.timestamp = _get_latest_timestamp(self.exp_path)

        all_predicted_labels = []
        all_final_states = []
        
        n_runs = self.n_runs
        self.n_runs = 0

        file_paths_to_load = [path.join(self.gen_traj_path, f"attractor_labels_{run_idx}.txt") for run_idx in range(n_runs)]
        file_paths = []

        for file_path in file_paths_to_load:
            if os.path.exists(file_path):
                file_paths.append(file_path)
                self.n_runs += 1
            else:
                print(f"[ utils/roa ] Will load {self.n_runs} runs since file {file_path} not found. ")
                break

        if not parallel:
            for file_path in tqdm(file_paths, desc="Loading final states"):
                labels, final_states = _load_attractor_labels(file_path)
               
                all_predicted_labels.append(labels)
                all_final_states.append(final_states)
        else: 
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = list(tqdm(pool.imap(_load_attractor_labels, file_paths), 
                                   total=len(file_paths), 
                                   desc="Loading final states"))

            for labels, final_states in results:
                all_predicted_labels.append(labels)
                all_final_states.append(final_states)
                

        self.attractor_labels = np.array(all_predicted_labels, dtype=np.int32).transpose(1, 0)
        self.final_states = np.array(all_final_states, dtype=np.float32).transpose(1, 0, 2)

        if self.n_runs != self._expected_n_runs:
            warnings.warn(f"Loaded {self.n_runs} runs instead of {self._expected_n_runs}")
        
        return self.attractor_labels, self.final_states
    
    def compute_attractor_probabilities(self, plot=False, verbose=True):
        self._setup_results_path()

        if verbose:
            print(f"[ utils/roa ] Computing attractor probabilities")
        if self.attractor_labels is None:
            raise ValueError("No attractor labels available")
        
        if self.labels_set is None or self.labels_array is None:
            raise ValueError("Labels set not computed. Run set_labels first.")
            
        n_points = len(self.start_points)
        
        self.label_probabilities = np.zeros((n_points, len(self.labels_set) + 1))
        
        invalid_label = self.labels_array[-1]
        
        # Compute probability of invalid label
        self.label_probabilities[:, -1] = np.sum(self.attractor_labels == invalid_label, axis=1) / self.n_runs

        # Compute probability of each attractor label
        for i, label in enumerate(self.labels_set):
            self.label_probabilities[:, i] = np.sum(self.attractor_labels == label, axis=1) / self.n_runs

        # Check that probabilities sum to 1 (allowing for floating point error)
        assert np.allclose(np.sum(self.label_probabilities, axis=1), 1.0, rtol=1e-5, atol=1e-6), "Probabilities do not sum to 1"
        
        if plot:
            self.plot_attractor_probabilities(verbose)
        
        return self.label_probabilities
    
    def predict_attractor_labels(self, plot=False, verbose=True):
        self._setup_results_path()

        if verbose:
            print(f"[ utils/roa ] Predicting attractor labels")
        invalid_label = self.labels_array[-1]

        if self.label_probabilities is None:
            self.compute_attractor_probabilities()
        
        if self.labels_array is None:
            raise ValueError("Labels array not computed. Call compute_attractor_labels first.")
        
        predicted_labels = np.full(len(self.start_points), invalid_label, dtype=np.int32)
        
        # Assign label with highest probability exceeding threshold
        for i in range(len(self.labels_array[:-1])):
            mask = self.label_probabilities[:, i] > self.attractor_prob_threshold
            predicted_labels[mask] = self.labels_array[i]

        self.predicted_labels = predicted_labels

        # Store indices of uncertain points - points with no probability exceeding threshold
        uncertain_indices = np.where(np.sum(self.label_probabilities > self.attractor_prob_threshold, axis=1) == 0)[0]
        self.uncertain_indices = uncertain_indices
        
        if plot:
            self.plot_predicted_attractor_labels(verbose)
        
        return predicted_labels
    
    def plot_attractor_probabilities(self, verbose=True):
        self._setup_results_path()

        if verbose:
            print(f"[ utils/roa ] Plotting attractor probabilities")
        
        if self.label_probabilities is None:
            raise ValueError("No attractor probabilities available")
        
        if self.start_points.shape[1] > 2:
            raise ValueError("Cannot plot probabilities for more than 2D start points")
        
        os.makedirs(self.results_path, exist_ok=True)
        
        # Plot probability of each attractor label
        for i, label in enumerate(self.labels_set):
            plt.figure(figsize=(10, 8))
            plt.scatter(self.start_points[:, 0], self.start_points[:, 1], 
                       c=self.label_probabilities[:, i], s=0.1, cmap='RdBu')
            plt.colorbar()
            plt.title(f'Probability of label {label}')
            plt.savefig(path.join(self.results_path, f"probability_{label}.png"))
            plt.close()

        # Plot probability of invalid label
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_points[:, 0], self.start_points[:, 1], 
                   c=self.label_probabilities[:, -1], s=0.1, cmap='RdBu')
        plt.colorbar()
        plt.title('Probability of invalid label')
        plt.savefig(path.join(self.results_path, f"probability_invalid.png"))
        plt.close()
    
    def plot_predicted_attractor_labels(self, verbose=True):
        self._setup_results_path()

        if verbose:
            print(f"[ utils/roa ] Plotting predicted attractor labels")

        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")
        
        if self.start_points.shape[1] > 2:
            raise ValueError("Cannot plot probabilities for more than 2D start points")
        
        invalid_label = self.labels_array[-1]
        
        os.makedirs(self.results_path, exist_ok=True)
        
        for label in self.labels_set:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.start_points[:, 0], self.start_points[:, 1], 
                       c=self.predicted_labels == label, s=0.1, cmap='viridis')
            plt.title(f'{label} Label Predictions')
            plt.savefig(path.join(self.results_path, f"predicted_{label}.png"))
            plt.close()

        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_points[:, 0], self.start_points[:, 1], 
                   c=self.predicted_labels == invalid_label, s=0.1, cmap='viridis')
        plt.title('Invalid Label Predictions')
        plt.savefig(path.join(self.results_path, f"predicted_invalid_label.png"))
        plt.close()

        if self.uncertain_indices is not None:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.start_points[self.uncertain_indices, 0], self.start_points[self.uncertain_indices, 1], 
                       c='red', s=0.1, cmap='viridis')
            plt.title('Uncertain points')
            plt.savefig(path.join(self.results_path, f"uncertain_points.png"))
            plt.close()

        print(f"[ utils/roa ] Attractor probabilities plotted in {self.results_path}")

    def plot_roas(self, verbose=True):
        self._setup_results_path()

        if verbose:
            print(f"[ utils/roa ] Plotting ROAs")
        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")
        
        if self.start_points.shape[1] > 2:
            raise ValueError("Cannot plot ROAs for more than 2D start points")
        
        labels_to_plot = self.predicted_labels.copy()
        labels_to_plot[labels_to_plot == self.invalid_label] = len(self.labels_set)
        
        plt.figure(figsize=(10, 8))
        
        plt.scatter(self.start_points[:, 0], self.start_points[:, 1], 
                   c=labels_to_plot, s=0.1, cmap='viridis')
        plt.colorbar()

        plt.title('RoAs')
        plt.savefig(path.join(self.results_path, f"roas.png"))
        plt.close()

    def compute_prediction_metrics(self, save=True):
        self._setup_results_path()

        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")
        
        if self.labels_array is None:
            raise ValueError("Labels array not computed. Call set_labels first.")
        
        if len(self.labels_array) == 3:
            invalid_label = self.labels_array[-1]
            
            fp = np.sum((self.predicted_labels == self.labels_array[0]) & (self.expected_labels == self.labels_array[1]))
            fn = np.sum((self.predicted_labels == self.labels_array[1]) & (self.expected_labels == self.labels_array[0]))
            tp = np.sum((self.predicted_labels == self.labels_array[1]) & (self.expected_labels == self.labels_array[1]))
            tn = np.sum((self.predicted_labels == self.labels_array[0]) & (self.expected_labels == self.labels_array[0]))

            tp_rate = tp / (tp + fn)
            tn_rate = tn / (tn + fp)
            fp_rate = fp / (fp + tn)
            fn_rate = fn / (fn + tp)

            accuracy = (tp + tn) / len(self.expected_labels)
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

            results = {
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "tp_rate": float(tp_rate),
                "tn_rate": float(tn_rate),
                "fp_rate": float(fp_rate),
                "fn_rate": float(fn_rate),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score)
            }

        else:
            # Multi-class classification metrics
            invalid_label = self.labels_array[-1]
            valid_labels = [label for label in self.labels_array if label != invalid_label]
            
            # Compute accuracy
            accuracy = np.sum(self.predicted_labels == self.expected_labels) / len(self.expected_labels)
            
            # Compute per-class metrics
            class_metrics = {}
            
            for label in valid_labels:
                # One-vs-rest approach
                true_positives = np.sum((self.predicted_labels == label) & (self.expected_labels == label))
                false_positives = np.sum((self.predicted_labels == label) & (self.expected_labels != label))
                false_negatives = np.sum((self.predicted_labels != label) & (self.expected_labels == label))
                true_negatives = np.sum((self.predicted_labels != label) & (self.expected_labels != label))
                # Handle division by zero
                micro_accuracy = (true_positives + true_negatives) / len(self.expected_labels)
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else np.nan
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else np.nan
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

                tp_rate = true_positives / (true_positives + false_negatives)
                tn_rate = true_negatives / (true_negatives + false_positives)
                fp_rate = false_positives / (false_positives + true_negatives)
                fn_rate = false_negatives / (false_negatives + true_positives)
                
                class_metrics[f"class_{label}"] = {
                    "true_positives": int(true_positives),
                    "false_positives": int(false_positives),
                    "true_negatives": int(true_negatives),
                    "false_negatives": int(false_negatives),
                    "tp_rate": float(tp_rate),
                    "tn_rate": float(tn_rate),
                    "fp_rate": float(fp_rate),
                    "fn_rate": float(fn_rate),
                    "accuracy": float(micro_accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1)
                }
            
            # Compute macro-averages
            macro_precision = np.mean([metrics["precision"] for metrics in class_metrics.values()])
            macro_recall = np.mean([metrics["recall"] for metrics in class_metrics.values()])
            macro_f1 = np.mean([metrics["f1_score"] for metrics in class_metrics.values()])
            
            # Create confusion matrix
            confusion_matrix = np.zeros((len(self.labels_array), len(self.labels_array)), dtype=np.int32)
            
            for i, true_label in enumerate(self.labels_array):
                for j, pred_label in enumerate(self.labels_array):
                    confusion_matrix[i, j] = np.sum(
                        (self.expected_labels == true_label) & (self.predicted_labels == pred_label)
                    )
            
            results = {
                "accuracy": float(accuracy),
                "class_metrics": class_metrics,
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "confusion_matrix": confusion_matrix.tolist(),
                "confusion_matrix_labels": self.labels_array.tolist()
            }

        if save:
            # Create results directory if it doesn't exist
            os.makedirs(self.results_path, exist_ok=True)
            
            # Save classification results to JSON file
            results_fpath = path.join(self.results_path, "classification_results.json")
            with open(results_fpath, "w") as f:
                json.dump(results, f, indent=4)
            
            print(f"[ utils/roa ] Classification results saved to {results_fpath}")

        print(f"\n[ utils/roa ] Classification results for {self.exp_path} | {self.n_runs} runs:\n")
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
        
    
        
        
        
        
        
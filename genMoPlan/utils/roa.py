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
from typing import Optional

from tqdm import tqdm

from genMoPlan.utils.json_args import JSONArgs
from genMoPlan.utils.trajectory import plot_trajectories

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

def _get_latest_timestamp(model_path, final_state_directory):
    if not os.path.exists(model_path) or len(os.listdir(model_path)) == 0:
        return None
    
    timestamps = [path.basename(d) for d in glob.glob(path.join(model_path, final_state_directory, "*"))]
    return max(timestamps)

class ROAEstimator:
    SUCCESS_COLOR_MAP = "tab10_r"
    FAILURE_COLOR_MAP = "tab20b"
    FP_COLOR_MAP = "bwr_r"
    FN_COLOR_MAP = "Wistia_r"
    SEPARATRIX_COLOR_MAP = "autumn_r"

    model_state_name: str = "best.pt"
    model_path: str = None
    gen_traj_path: str = None
    results_path: str = None
    traj_plot_path: str = None
    method_name: str = None
    _timestamp: str = None
    
    dataset: str = None
    model: torch.nn.Module = None
    model_args: JSONArgs = None
    batch_size: int = None
    horizon_length: int = None
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
    separatrix_indices: np.ndarray = None

    # Classification
    inference_params: dict = None
    attractor_labels: np.ndarray = None
    label_probabilities: np.ndarray = None
    invalid_label: int = None
    attractors: dict = None
    attractor_dist_threshold: float = None
    attractor_prob_threshold: float = None
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
    ):
        self.dataset = dataset
        self.model_path = model_path
        self.verbose = verbose
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device

        self._orig_n_runs = n_runs

        self._load_model(model_state_name)
        self._load_params()

       
    def _load_model(self, model_state_name: str):
        from genMoPlan.utils import load_model

        self.model, self.model_args = load_model(self.model_path, self.device, model_state_name, verbose=self.verbose, strict=False)
    
    def _load_params(self):
        from genMoPlan.utils import load_inference_params, get_method_name

        self.inference_params = load_inference_params(self.dataset)

        self.n_runs = self._orig_n_runs if self._orig_n_runs is not None else self.inference_params["n_runs"]
        self._expected_n_runs = self.n_runs
        self.batch_size = self.batch_size if self.batch_size is not None else self.inference_params["batch_size"]

        self.attractor_dist_threshold = self.inference_params["attractor_dist_threshold"]
        self.attractor_prob_threshold = self.inference_params["attractor_prob_threshold"]
        self.attractors = self.inference_params["attractors"]
        self.invalid_label = self.inference_params["invalid_label"]
        self.max_path_length = self.inference_params["max_path_length"]
        self.final_state_directory = self.inference_params["final_state_directory"]
        self.labels_set = set(list(self.attractors.values()))
        self.labels_array = np.array([*self.labels_set, self.invalid_label])    

        self.method_name = get_method_name(self.model_args)
        self.conditional_sample_kwargs = self.inference_params[self.method_name] if self.method_name in self.inference_params else {}

        self.horizon_length = self.model_args["horizon_length"]

    def set_attractor_dist_threshold(self, attractor_dist_threshold: float):
        self.attractor_dist_threshold = attractor_dist_threshold
        self.inference_params["attractor_dist_threshold"] = attractor_dist_threshold

    def set_attractor_prob_threshold(self, attractor_prob_threshold: float):
        self.attractor_prob_threshold = attractor_prob_threshold
        self.inference_params["attractor_prob_threshold"] = attractor_prob_threshold

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        self.inference_params["batch_size"] = batch_size

    def set_horizon_and_max_path_lengths(self, horizon_length: Optional[int] = None, max_path_length: Optional[int] = None, num_inference_steps: Optional[int] = None):
        if horizon_length is not None:
            self.horizon_length = horizon_length
            self.inference_params["horizon_length"] = horizon_length

        if max_path_length is None and num_inference_steps is None:
            return
        
        if max_path_length is not None and num_inference_steps is not None:
            raise ValueError("Cannot set both max_path_length and num_inference_steps")
        
        if max_path_length is not None:
            self.max_path_length = max_path_length
            self.inference_params["max_path_length"] = max_path_length
        else:
            self.max_path_length = num_inference_steps * self.horizon_length
            self.inference_params["max_path_length"] = self.max_path_length

    def reset(self, for_analysis: bool = False):
        """
        Resets the ROAEstimator for new runs while retaining the ground truth data

        Args:
            for_analysis: If True, only the analysis results will be reset but the run data will be retained.
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
        self.start_points = None
        self.expected_labels = None
        self.n_runs = self._orig_n_runs
        
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

    def load_ground_truth(self):
        if self.verbose:
            print('[ scripts/estimate_roa ] Loading ground truth')
        roa_labels_fpath = path.join("data_trajectories", self.dataset, "roa_labels.txt")

        start_points = []
        expected_labels = []
        
        if os.path.exists(roa_labels_fpath):
            with open(roa_labels_fpath, "r") as f:
                for line in f:
                    line_data = line.strip().split(' ')[1:]
                    start_points.append([np.float32(line_data[0]), np.float32(line_data[1])])
                    expected_labels.append(int(line_data[2]))
        else:
            raise FileNotFoundError(f"File {roa_labels_fpath} not found")
        
        self.start_points = np.array(start_points, dtype=np.float32)
        self.expected_labels = np.array(expected_labels, dtype=np.int32)

        if self.num_batches is not None:
            self.set_batch_size(self.start_points.shape[0] // self.num_batches)

        if self.verbose:
            print(f"[ utils/roa ] Loaded {self.start_points.shape[0]} ground truth data points")
     
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
    
    @property
    def timestamp(self):
        return self._timestamp
    
    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value
        self.gen_traj_path = path.join(self.model_path, self.final_state_directory, self._timestamp)

        if not os.path.exists(self.gen_traj_path):
            os.makedirs(self.gen_traj_path)

        self._save_inference_params(self.gen_traj_path)
        
    
    def _generate_single_run_trajs(
            self,
            run_idx,
            compute_labels,
            discard_trajectories,
            save,
        ):
        from genMoPlan.utils import generate_trajectories

        results = generate_trajectories(
            self.model, 
            self.model_args, 
            self.start_points, 
            max_path_length=self.max_path_length, 
            device=self.device,
            horizon_length=self.horizon_length,
            verbose=self.verbose, 
            batch_size=self.batch_size, 
            only_return_final_states=discard_trajectories,
            conditional_sample_kwargs=self.conditional_sample_kwargs,
            post_process_fns=self.inference_params["post_process_fns"],
            post_process_fn_kwargs=self.inference_params["post_process_fn_kwargs"],
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

            attractor_labels = get_trajectory_attractor_labels(final_states, self.attractors, self.attractor_dist_threshold, self.invalid_label, verbose=self.verbose)
            self.attractor_labels.append(attractor_labels)

        if save:
            self._save_single_run_data(run_idx, final_states)
                
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
    
    def _save_single_run_data(self, run_idx, final_states):
        file_path = path.join(self.gen_traj_path, f"attractor_labels_{run_idx}.txt")
        with open(file_path, "w") as f:
            f.write("# ")
            for i in range(self.start_points.shape[1]):
                f.write(f"start_{i} ")
            for i in range(final_states.shape[1]):
                f.write(f"final_{i} ")
            f.write("\n")
            
            for start_point_idx in range(self.start_points.shape[0]):
                for i in range(self.start_points.shape[1]):
                    f.write(f"{self.start_points[start_point_idx, i]} ")
                    
                for i in range(final_states.shape[1]):
                    f.write(f"{final_states[start_point_idx, i]} ")
                
                f.write("\n")

        print(f"[ utils/roa ] Attractor labels saved in {file_path}")
    
    def generate_trajectories(
            self,
            compute_labels: bool = True,
            discard_trajectories: bool = True, 
            save: bool = False,
        ):
        from genMoPlan.utils.progress import ETAIterator

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
            if self.verbose:
                print(f"[ utils/roa ] Run {run_idx+1}/{start_idx + n_runs_to_generate} (Remaining Time: {eta_iter.eta_formatted})")
            
            self._generate_single_run_trajs(
                run_idx, compute_labels, discard_trajectories, save
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
    
    def compute_attractor_labels(self):
        from genMoPlan.utils import get_trajectory_attractor_labels

        if self.verbose:
            print(f"[ utils/roa ] Computing attractor labels")
        if self.final_states is None:
            raise ValueError("No final states available.")
        
        n, n_runs, dim = self.final_states.shape
        reshaped_final_states = self.final_states.reshape(-1, dim)
        
        reshaped_labels = get_trajectory_attractor_labels(
            reshaped_final_states, self.attractors, self.attractor_dist_threshold, self.invalid_label, verbose=self.verbose
        )
        
        self.attractor_labels = reshaped_labels.reshape(n, n_runs)
        self.attractor_labels = np.array(self.attractor_labels, dtype=np.int32)
        
        return self.attractor_labels
    
    def save_final_states(self):
        if self.verbose:
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

    def save_generated_trajectories(self):
        if self.verbose:
            print(f"[ utils/roa ] Saving generated trajectories")
        if self.trajectories is None:
            raise ValueError("No generated trajectories available")
        
        
    
    def load_final_states(self, timestamp: str = None, parallel=True):
        if timestamp is None:
            self.timestamp = _get_latest_timestamp(self.model_path, self.final_state_directory)

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

        if self.n_runs == 0:
            raise ValueError("No final states found")

        if not parallel:
            iter = tqdm(file_paths, desc="Loading final states") if self.verbose else file_paths
            for file_path in iter:
                labels, final_states = _load_attractor_labels(file_path)
               
                all_predicted_labels.append(labels)
                all_final_states.append(final_states)
        else: 
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                if self.verbose:
                    iter = tqdm(pool.imap(_load_attractor_labels, file_paths), 
                                   total=len(file_paths), 
                                   desc="Loading final states")
                else:
                    iter = pool.imap(_load_attractor_labels, file_paths)

                results = list(iter)

            for labels, final_states in results:
                all_predicted_labels.append(labels)
                all_final_states.append(final_states)
                

        self.attractor_labels = np.array(all_predicted_labels, dtype=np.int32).transpose(1, 0)
        self.final_states = np.array(all_final_states, dtype=np.float32).transpose(1, 0, 2)

        if self.n_runs != self._expected_n_runs:
            warnings.warn(f"Loaded {self.n_runs} runs instead of {self._expected_n_runs}")
        
        return self.attractor_labels, self.final_states
    
    def compute_attractor_probabilities(self, plot=False):
        if self.verbose:
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
            self.plot_attractor_probabilities()
        
        return self.label_probabilities
    
    def _save_predicted_labels(self):
        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")
        
        self._setup_results_path()

        file_path = path.join(self.results_path, f'predicted_attractor_labels.txt')

        with open(file_path, "w") as f:
            f.write("# ")
            for i in range(self.start_points.shape[1]):
                f.write(f"start_{i} ")
            f.write("label\n")

            for start_point_idx in range(self.start_points.shape[0]):
                for i in range(self.start_points.shape[1]):
                    f.write(f"{self.start_points[start_point_idx, i]} ")
                
                f.write(f"{self.predicted_labels[start_point_idx]} ")
                
                f.write("\n")
        if self.verbose:
            print(f"[ utils/roa ] Saved predicted labels to {file_path}")
    
    def load_predicted_labels(self, results_path: str):
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
                start_points = np.array(data[1:-1], dtype=np.float32)
                predicted_label = int(data[-1])
                self.predicted_labels.append(predicted_label)
        
        self.predicted_labels = np.array(self.predicted_labels, dtype=np.int32)
        self.separatrix_indices = np.where(self.predicted_labels == self.invalid_label)[0]
        
        if self.verbose:
            print(f"[ utils/roa ] Loaded {len(self.predicted_labels)} predicted labels")
    
    def predict_attractor_labels(self, save=True, plot=False):
        if self.verbose:
            print(f"[ utils/roa ] Predicting attractor labels")
        invalid_label = self.labels_array[-1]

        if self.label_probabilities is None:
            self.compute_attractor_probabilities()
        
        if self.labels_array is None:
            raise ValueError("Labels array not computed. Call compute_attractor_labels first.")
        
        predicted_labels = np.full(len(self.start_points), invalid_label, dtype=np.int32)
        
        # Assign label with highest probability exceeding threshold
        for i in range(len(self.labels_array[:-1])):
            mask = self.label_probabilities[:, i] >= self.attractor_prob_threshold
            predicted_labels[mask] = self.labels_array[i]

        self.predicted_labels = predicted_labels

        # Store indices of uncertain points - points with no probability exceeding threshold
        uncertain_indices = np.where(np.sum(self.label_probabilities >= self.attractor_prob_threshold, axis=1) == 0)[0]
        self.uncertain_indices = uncertain_indices

        self.separatrix_indices = np.where(predicted_labels == invalid_label)[0]
        
        if self.verbose:
            print(f"[ utils/roa ] Found {len(uncertain_indices)} uncertain points")
            print(f"[ utils/roa ] Total separatrix points: {len(self.separatrix_indices)}")

        if save:
            self._save_predicted_labels()
        
        if plot:
            self.plot_predicted_attractor_labels()
        
        return predicted_labels
    
    def plot_trajectories(self):
        self._setup_traj_plot_path()

        if self.verbose:
            print(f"[ utils/roa ] Plotting trajectories")

        if self.trajectories is None:
            raise ValueError("No trajectories available")
        
        if self.start_points.shape[1] > 2:
            raise ValueError("Cannot plot trajectories for more than 2D start points")
        
        trajectories = np.concatenate(self.trajectories, axis=0)

        image_name = f"trajectories_{len(self.start_points)}_{self.n_runs}.png"
        image_path = path.join(self.traj_plot_path, image_name)
        
        plot_trajectories(trajectories, image_path, self.verbose)
        
        if self.verbose:
            print(f"[ utils/roa ] Trajectories plotted in {image_path}")
        
    def plot_attractor_probabilities(self):
        self._setup_results_path()

        if self.verbose:
            print(f"[ utils/roa ] Plotting attractor probabilities")
        
        if self.label_probabilities is None:
            raise ValueError("No attractor probabilities available")
        
        if self.start_points.shape[1] > 2:
            raise ValueError("Cannot plot probabilities for more than 2D start points")
        
        os.makedirs(self.results_path, exist_ok=True)
        
        for i, label in enumerate(self.labels_set):
            plt.figure(figsize=(10, 8))
            plt.scatter(self.start_points[:, 0], self.start_points[:, 1], 
                    c=self.label_probabilities[:, i], s=0.1, cmap='RdBu')
            plt.colorbar()
            plt.title(f'Probability of label {label}')
            plt.savefig(path.join(self.results_path, f"probability_{label}.png"))
            plt.close()

        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_points[:, 0], self.start_points[:, 1], 
                c=self.label_probabilities[:, -1], s=0.1, cmap='RdBu')
        plt.colorbar()
        plt.title('Probability of invalid label')
        plt.savefig(path.join(self.results_path, f"probability_invalid.png"))
        plt.close()
    
    def plot_predicted_attractor_labels(self):
        self._setup_results_path()

        if self.verbose:
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
                       c='red', s=0.1)
            plt.title('Uncertain points')
            plt.savefig(path.join(self.results_path, f"uncertain_points.png"))
            plt.close()

        else:
            print(f"[ utils/roa ] No uncertain points found or loaded")

        if self.verbose:    
            print(f"[ utils/roa ] Attractor probabilities plotted in {self.results_path}")

    def plot_roas(self, plot_separatrix=True):
        self._setup_results_path()

        if self.verbose:
            print(f"[ utils/roa ] Plotting ROAs")
        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")
        
        if self.start_points.shape[1] > 2:
            raise ValueError("Cannot plot ROAs for more than 2D start points")
        
        labels_to_plot = self.predicted_labels.copy()
        labels_to_plot[labels_to_plot == self.invalid_label] = len(self.labels_set)

        success_mask = labels_to_plot == self.labels_array[1]
        failure_mask = labels_to_plot == self.labels_array[0]
        separatrix_mask = labels_to_plot == len(self.labels_set)
        
        plt.figure(figsize=(10, 8))
        
        plt.scatter(self.start_points[success_mask, 0], self.start_points[success_mask, 1], 
                   c=labels_to_plot[success_mask], s=0.1, cmap=self.SUCCESS_COLOR_MAP)
        plt.scatter(self.start_points[failure_mask, 0], self.start_points[failure_mask, 1], 
                   c=labels_to_plot[failure_mask], s=0.1, cmap=self.FAILURE_COLOR_MAP)
        plt.scatter(self.start_points[separatrix_mask, 0], self.start_points[separatrix_mask, 1], 
                   c=labels_to_plot[separatrix_mask], s=0.1, cmap=self.SEPARATRIX_COLOR_MAP)

        plt.title('RoAs')
        plt.savefig(path.join(self.results_path, f"roas.png"))
        plt.close()

        if plot_separatrix:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.start_points[self.separatrix_indices, 0], self.start_points[self.separatrix_indices, 1], 
                       c='red', s=0.1)
            plt.title('Separatrix')
            plt.savefig(path.join(self.results_path, f"separatrix.png"))
            plt.close()

    def compute_classification_results(self, save=True):
        self._setup_results_path()

        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")
        
        if self.labels_array is None:
            raise ValueError("Labels array not computed. Call set_labels first.")
        
        fp = np.sum((self.predicted_labels == self.labels_array[1]) & (self.expected_labels == self.labels_array[0]))
        fn = np.sum((self.predicted_labels == self.labels_array[0]) & (self.expected_labels == self.labels_array[1]))
        tp = np.sum((self.predicted_labels == self.labels_array[1]) & (self.expected_labels == self.labels_array[1]))
        tn = np.sum((self.predicted_labels == self.labels_array[0]) & (self.expected_labels == self.labels_array[0]))

        # Compute masks for true positives, true negatives, false positives, and false negatives
        self.fp_mask = (self.predicted_labels == self.labels_array[1]) & (self.expected_labels == self.labels_array[0])
        self.fn_mask = (self.predicted_labels == self.labels_array[0]) & (self.expected_labels == self.labels_array[1])
        self.tp_mask = (self.predicted_labels == self.labels_array[1]) & (self.expected_labels == self.labels_array[1])
        self.tn_mask = (self.predicted_labels == self.labels_array[0]) & (self.expected_labels == self.labels_array[0])

        tp_rate = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        tn_rate = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

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
        
    def plot_classification_results(self):
        self._setup_results_path()

        if self.verbose:
            print(f"[ utils/roa ] Plotting classification results")

        if self.predicted_labels is None:
            raise ValueError("No predicted labels available")

        if self.tp_mask is None or self.tn_mask is None or self.fp_mask is None or self.fn_mask is None:
            raise ValueError("Classification results not computed. Call compute_prediction_metrics first.")
        
        if self.start_points.shape[1] > 2:
            raise ValueError("Cannot plot classification results for more than 2D start points")

        # Plot true positives
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_points[self.tp_mask, 0], self.start_points[self.tp_mask, 1], 
                   c=self.predicted_labels[self.tp_mask], s=0.1, cmap=self.SUCCESS_COLOR_MAP)
        plt.title('True Positives')
        plt.xlim(self.start_points[:, 0].min(), self.start_points[:, 0].max())
        plt.ylim(self.start_points[:, 1].min(), self.start_points[:, 1].max())
        plt.savefig(path.join(self.results_path, "true_positives.png"))
        plt.close()
        
        # Plot true negatives
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_points[self.tn_mask, 0], self.start_points[self.tn_mask, 1], 
                   c=self.predicted_labels[self.tn_mask], s=0.1, cmap=self.FAILURE_COLOR_MAP)
        plt.title('True Negatives')
        plt.xlim(self.start_points[:, 0].min(), self.start_points[:, 0].max())
        plt.ylim(self.start_points[:, 1].min(), self.start_points[:, 1].max())
        plt.savefig(path.join(self.results_path, "true_negatives.png"))
        plt.close()
        
        # Plot false positives
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_points[self.fp_mask, 0], self.start_points[self.fp_mask, 1], 
                   c=self.predicted_labels[self.fp_mask], s=0.1, cmap=self.FP_COLOR_MAP)
        plt.title('False Positives')
        plt.xlim(self.start_points[:, 0].min(), self.start_points[:, 0].max())
        plt.ylim(self.start_points[:, 1].min(), self.start_points[:, 1].max())
        plt.savefig(path.join(self.results_path, "false_positives.png"))
        plt.close()
        
        # Plot false negatives
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_points[self.fn_mask, 0], self.start_points[self.fn_mask, 1], 
                   c=self.predicted_labels[self.fn_mask], s=0.1, cmap=self.FN_COLOR_MAP)
        plt.title('False Negatives')
        plt.xlim(self.start_points[:, 0].min(), self.start_points[:, 0].max())
        plt.ylim(self.start_points[:, 1].min(), self.start_points[:, 1].max())
        plt.savefig(path.join(self.results_path, "false_negatives.png"))
        plt.close()

        # Combined plots with all incorrect classifications
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_points[self.fp_mask, 0], self.start_points[self.fp_mask, 1], 
                   c=self.predicted_labels[self.fp_mask], s=0.1, cmap=self.FP_COLOR_MAP, label='False Positives')
        plt.scatter(self.start_points[self.fn_mask, 0], self.start_points[self.fn_mask, 1], 
                   c=self.predicted_labels[self.fn_mask], s=0.1, cmap=self.FN_COLOR_MAP, label='False Negatives')
        plt.title('Incorrect Classifications')
        plt.xlim(self.start_points[:, 0].min(), self.start_points[:, 0].max())
        plt.ylim(self.start_points[:, 1].min(), self.start_points[:, 1].max())
        plt.savefig(path.join(self.results_path, "incorrect_classifications.png"))
        plt.close()
        
        # Combined plot with all classifications
        plt.figure(figsize=(10, 8))
        plt.scatter(self.start_points[self.tp_mask, 0], self.start_points[self.tp_mask, 1], 
                   c=self.predicted_labels[self.tp_mask], s=0.1, cmap=self.SUCCESS_COLOR_MAP, label='True Positives')
        plt.scatter(self.start_points[self.tn_mask, 0], self.start_points[self.tn_mask, 1], 
                   c=self.predicted_labels[self.tn_mask], s=0.1, cmap=self.FAILURE_COLOR_MAP, label='True Negatives')
        plt.scatter(self.start_points[self.fp_mask, 0], self.start_points[self.fp_mask, 1], 
                   c=self.predicted_labels[self.fp_mask], s=0.1, cmap=self.FP_COLOR_MAP, label='False Positives')
        plt.scatter(self.start_points[self.fn_mask, 0], self.start_points[self.fn_mask, 1], 
                   c=self.predicted_labels[self.fn_mask], s=0.1, cmap=self.FN_COLOR_MAP, label='False Negatives')
        plt.scatter(self.start_points[self.separatrix_indices, 0], self.start_points[self.separatrix_indices, 1], 
                   c=self.predicted_labels[self.separatrix_indices], s=0.1, cmap=self.SEPARATRIX_COLOR_MAP, label='Separatrix')
        plt.title('Classification Results')
        plt.xlim(self.start_points[:, 0].min(), self.start_points[:, 0].max())
        plt.ylim(self.start_points[:, 1].min(), self.start_points[:, 1].max())
        plt.savefig(path.join(self.results_path, "classification_results.png"))
        plt.close()

        if self.verbose:
            print(f"[ utils/roa ] Classification result plots saved in {self.results_path}")
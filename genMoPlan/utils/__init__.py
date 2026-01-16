import os
import glob
from argparse import Namespace

from .arrays import *
from .class_loader import *
from .constants import *
from .data_processing import *
from .generation_result import *
from .json_args import *
from .manifold import *
from .model import *
from .parallel import *
from .params import *
from .paths import *
from .progress import *
from .setup import *
from .timer import *
from .trainer import *
from .trajectory import *
from .trajectory_generator import *
from .visualization import *


def generate_timestamp():
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def has_best_pt(directory):
    return os.path.exists(os.path.join(directory, "best.pt"))

def _expand_wildcard_path(path, no_best_pt=False):
    if "*" in path:
        # Handle wildcard paths
        # Use glob.glob directly with the path containing the wildcard
        all_dirs = glob.glob(path)
        
        # Filter directories that contain best.pt
        valid_dirs = [d for d in all_dirs if os.path.isdir(d) and (has_best_pt(d) or no_best_pt)]
        
        # Add valid directories to expanded paths
        return valid_dirs
    else:
        # Direct path without wildcard
        if os.path.exists(path) and (has_best_pt(path) or no_best_pt):
            return [path]
        else:
            return []

def expand_model_paths(model_paths, no_best_pt=False):
    if isinstance(model_paths, str):
        model_paths = [model_paths]

    expanded_paths = []

    print(f"Searching for model paths with the working directory: {os.getcwd()}")
    
    for path in model_paths:
        expanded_path = _expand_wildcard_path(path, no_best_pt=no_best_pt)

        if len(expanded_path) == 0:
            print(f"Warning: Path {path} does not exist or does not contain best.pt")
        else:
            expanded_paths.extend(expanded_path)

    print(f"Found {len(expanded_paths)} valid model paths")
    if len(expanded_paths) == 0:
        raise ValueError("No valid model paths found")

    for model_path in expanded_paths:
        print(f"  - {model_path}")

    input("Press Enter to continue...")
    
    return expanded_paths

def get_non_angular_indices(angle_indices: List[int], dimensions: int) -> List[int]:
    return [i for i in range(dimensions) if i not in angle_indices]



def load_roa_labels(
    dataset: str, 
) -> np.ndarray:
    roa_labels_fpath = path.join(get_data_trajectories_path(), dataset, "roa_labels.txt")

    start_states = []
    expected_labels = []

    if os.path.exists(roa_labels_fpath):
            with open(roa_labels_fpath, "r") as f:
                for line in f:
                    line_data = line.strip().split(',')
                    start_states.append([np.float32(value) for value in line_data[:-1]])
                    expected_labels.append(int(float(line_data[-1])))
    else:
        raise FileNotFoundError(f"File {roa_labels_fpath} not found")

    return np.array(start_states, dtype=np.float32), np.array(expected_labels, dtype=np.int32)



def query_roa_labels_for_start_points(query_points: np.ndarray, all_start_states: np.ndarray, all_roa_labels: np.ndarray, max_distance: float = 1e-2) -> np.ndarray:
    from scipy.spatial import cKDTree

    kdtree = cKDTree(all_start_states)
    distances, indices = kdtree.query(query_points)

    if distances.max() > max_distance:
        raise ValueError(f"Maximum distance between query points and all start states is greater than {max_distance}. Some query points are not in the given start states.")

    return all_roa_labels[indices]

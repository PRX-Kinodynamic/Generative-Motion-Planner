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
        valid_dirs = [
            d for d in all_dirs if os.path.isdir(d) and (has_best_pt(d) or no_best_pt)
        ]

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


def load_test_set(
    dataset: str,
    state_dim: int,
) -> tuple:
    """
    Load test_set.txt containing start states, ground truth final states, and labels.

    The file format is CSV with columns:
    - start_state (state_dim columns)
    - ground_truth_final_state (state_dim columns)
    - label (1 column)

    Args:
        dataset: Name of the dataset
        state_dim: Dimension of the state space

    Returns:
        Tuple of (start_states, ground_truth_final_states, labels)
        - start_states: np.ndarray of shape (N, state_dim)
        - ground_truth_final_states: np.ndarray of shape (N, state_dim)
        - labels: np.ndarray of shape (N,) with int32 dtype

    Raises:
        FileNotFoundError: If test_set.txt is not found
        ValueError: If column count doesn't match expected (2 * state_dim + 1)
    """
    test_set_fpath = path.join(get_data_trajectories_path(), dataset, "test_set.txt")

    if not os.path.exists(test_set_fpath):
        raise FileNotFoundError(f"File {test_set_fpath} not found")

    # Use efficient np.loadtxt for loading
    data = np.loadtxt(test_set_fpath, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]

    expected_cols = 2 * state_dim + 1
    if data.shape[1] != expected_cols:
        raise ValueError(
            f"test_set.txt has {data.shape[1]} columns, expected {expected_cols} "
            f"(2 * state_dim + 1 = 2 * {state_dim} + 1)"
        )

    start_states = data[:, :state_dim]
    ground_truth_final_states = data[:, state_dim : 2 * state_dim]
    labels = data[:, -1].astype(np.int32)

    return start_states, ground_truth_final_states, labels

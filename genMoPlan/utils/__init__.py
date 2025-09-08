import os
import glob


from .arrays import *
from .class_loader import *
from .data_processing import *
from .json_args import *
from .manifold import *
from .model import *
from .parallel import *
from .params import *
from .paths import *
from .progress import *
from .setup import *
from .timer import *
from .training import *
from .trajectory import *



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

def get_dataset_config(dataset: str, ):
    dataset = dataset.replace("-", "_")
    config = f'config.{dataset}'
    print(f"[ utils ] Reading config: {config}:{dataset}")
    module = importlib.import_module(config)
    params = getattr(module, "base")["base"].copy()
    return params
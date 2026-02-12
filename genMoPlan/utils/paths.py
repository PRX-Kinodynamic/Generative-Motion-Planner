import json
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load variables from a .env file if present
load_dotenv()


def get_data_trajectories_path(dataset: str = None) -> str:
    """Return the path to the data_trajectories directory.

    The path is determined by the DATA_TRAJECTORIES_PATH environment variable. If the
    variable is not set, the function falls back to the default relative path
    'data_trajectories'.
    """
    if dataset is None:
        return os.getenv("DATA_TRAJECTORIES_PATH", "data_trajectories")
    else:
        return os.path.join(os.getenv("DATA_TRAJECTORIES_PATH", "data_trajectories"), dataset)

# ---------------------------- experiments ----------------------------#
def get_experiments_path() -> str:
    """Return the path to the experiments directory.

    The path is determined by the EXPERIMENTS_PATH environment variable. If the
    variable is not set, the function falls back to the default relative path
    'experiments'.
    """
    return os.getenv("EXPERIMENTS_PATH", "experiments") 


def mkdir(savepath):
    """
    returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False


# ---------------------------- dataset description ----------------------------#


def load_dataset_description(dataset: str) -> Optional[Dict[str, Any]]:
    """Load dataset_description.json for a dataset if it exists.

    Args:
        dataset: Name of the dataset

    Returns:
        Parsed JSON as dict, or None if file doesn't exist
    """
    dataset_path = get_data_trajectories_path(dataset)
    description_path = os.path.join(dataset_path, "dataset_description.json")

    if not os.path.exists(description_path):
        return None

    with open(description_path, "r") as f:
        return json.load(f)


def get_achieved_bounds(
    dataset: str, state_names: List[str]
) -> Tuple[List[float], List[float]]:
    """
    Extract achieved_bounds from dataset_description.json as min/max lists.

    Args:
        dataset: Name of the dataset
        state_names: Ordered list of state dimension names (e.g., ["x", "theta", "x_dot", "theta_dot"])

    Returns:
        Tuple of (mins, maxs) lists ordered by state_names

    Raises:
        FileNotFoundError: If dataset_description.json doesn't exist
        KeyError: If required keys are missing from the JSON
    """
    description = load_dataset_description(dataset)
    if description is None:
        dataset_path = get_data_trajectories_path(dataset)
        raise FileNotFoundError(
            f"dataset_description.json not found at {dataset_path}. "
            "This file is required for normalization bounds."
        )

    if "achieved_bounds" not in description:
        raise KeyError(
            f"'achieved_bounds' key not found in dataset_description.json for {dataset}. "
            "This key is required for normalization bounds."
        )

    achieved = description["achieved_bounds"]

    mins = []
    maxs = []
    for name in state_names:
        if name not in achieved:
            raise KeyError(
                f"State '{name}' not found in achieved_bounds for {dataset}. "
                f"Available states: {list(achieved.keys())}"
            )
        mins.append(achieved[name]["min"])
        maxs.append(achieved[name]["max"])

    return mins, maxs
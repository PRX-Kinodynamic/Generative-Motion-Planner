import os
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
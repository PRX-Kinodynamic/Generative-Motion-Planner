import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from typing import Callable

from .trajectory import get_fnames_to_load



def _load_trajectory(traj_path: str) -> np.ndarray:
    return np.loadtxt(traj_path, delimiter=',')

def generate_roa_labels(dataset_path: str, success_criteria: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    traj_path = os.path.join(dataset_path, 'trajectories')
    traj_files = get_fnames_to_load(dataset_path)
    traj_paths = [os.path.join(traj_path, traj_file) for traj_file in traj_files]
    with Pool() as pool:
        traj_data = list(tqdm(pool.imap(_load_trajectory, traj_paths), total=len(traj_paths), desc="Loading trajectories"))
    start_states = np.array([traj[0] for traj in traj_data])
    final_states = np.array([traj[-1] for traj in traj_data])
    labels = np.array(success_criteria(final_states))
    
    roa_labels = np.zeros((start_states.shape[0], start_states.shape[1] + 1))
    roa_labels[:, :-1] = start_states
    roa_labels[:, -1] = labels
    np.savetxt(os.path.join(dataset_path, 'roa_labels.txt'), roa_labels, delimiter=',', fmt='%.6f')
    print(f"[ utils/dataset ] ROA labels generated and saved to {os.path.join(dataset_path, 'roa_labels.txt')}")


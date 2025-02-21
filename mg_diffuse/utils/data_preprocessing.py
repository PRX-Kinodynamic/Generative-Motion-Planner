import numpy as np
from tqdm import tqdm

from typing import List

def handle_trajectory_angle_wraparound(augment_new_state_data: bool, original_trajectory: np.ndarray) -> List[np.ndarray]:
    trajectory = original_trajectory.copy()
    angles = trajectory[:, 0]

    # Compute the difference between successive angle values
    angle_diffs = np.diff(angles)

    # Determine corrections: if the difference is too large, correct by ±2π
    corrections = np.zeros_like(angles)
    corrections[1:] = np.where(angle_diffs >= np.pi, -2 * np.pi, np.where(angle_diffs <= -np.pi, 2 * np.pi, 0))

    # Compute the cumulative correction for each state
    corrections = np.cumsum(corrections)

    # Apply the corrections to the original angles
    trajectory[:, 0] += corrections

    transformed_trajectories = [trajectory]

    if augment_new_state_data:
        # Check if the entire trajectory remains within [0, π]
        if np.all((original_trajectory[:, 0] >= 0) & (original_trajectory[:, 0] <= np.pi)):
            new_trajectory = trajectory.copy()
            new_trajectory[:, 0] -= 2 * np.pi
            transformed_trajectories.append(new_trajectory)

        # Check if the entire trajectory remains within [-π, 0]
        if np.all((original_trajectory[:, 0] >= -np.pi) & (original_trajectory[:, 0] <= 0)):
            new_trajectory = trajectory.copy()
            new_trajectory[:, 0] += 2 * np.pi
            transformed_trajectories.append(new_trajectory)

    return transformed_trajectories

def handle_angle_wraparound(trajectories, parallel=True, augment_new_state_data: bool = False, **kwargs):
    print(f"[ utils/data_preprocessing ] Handling angle wraparound")

    if not parallel:
        updated_results = []
        for trajectory in tqdm(trajectories):
            updated_results.append(handle_trajectory_angle_wraparound(augment_new_state_data, trajectory))
    else:
        import multiprocessing as mp
        from functools import partial

        # read trajectories in parallel with tqdm progress bar
        with mp.Pool(mp.cpu_count()) as pool:

            handle_trajectory_angle_wraparound_partial = partial(
                handle_trajectory_angle_wraparound, augment_new_state_data
            )

            updated_results = list(
                tqdm(
                    pool.imap(handle_trajectory_angle_wraparound_partial, trajectories),
                    total=len(trajectories),
                )
            )

    updated_trajectories = []

    for updated_result in updated_results:
        updated_trajectories.extend(updated_result)

    return np.array(updated_trajectories)

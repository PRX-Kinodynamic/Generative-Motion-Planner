import numpy as np
from tqdm import tqdm

def handle_trajectory_angle_wraparound(original_trajectory):
    correction = 0  # the value of correction is either 2pi or -2pi

    trajectory = original_trajectory.copy()

    for state_idx in range(1, trajectory.shape[0]):

        if original_trajectory[state_idx, 0] - original_trajectory[state_idx - 1, 0] >= np.pi:
            correction += -2 * np.pi

        elif original_trajectory[state_idx, 0] - original_trajectory[state_idx - 1, 0] <= -np.pi:
            correction += 2 * np.pi

        trajectory[state_idx, 0] += correction

    return trajectory

def handle_angle_wraparound(trajectories, parallel=True):
    """
        handle angle wraparound
    """
    print(f'[ utils/data_preprocessing ] Handling angle wraparound')
    updated_trajectories = []
    if not parallel:
        for trajectory in tqdm(trajectories):
            updated_trajectories.append(handle_trajectory_angle_wraparound(trajectory))
    else:
        import multiprocessing as mp

        # read trajectories in parallel with tqdm progress bar
        with mp.Pool(mp.cpu_count()) as pool:
            updated_trajectories = list(tqdm(pool.imap(handle_trajectory_angle_wraparound, trajectories), total=len(trajectories)))

    # for trajectory in updated_trajectories:
    #     for state_idx in range(1, trajectory.shape[0]):
    #         if abs(trajectory[state_idx, 0] - trajectory[state_idx - 1, 0]) >= np.pi:
    #             print('Error: angle wraparound not handled correctly')
    #             breakpoint()


    return np.array(updated_trajectories)
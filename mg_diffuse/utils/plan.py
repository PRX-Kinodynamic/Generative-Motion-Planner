from functools import partial
import os

import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count

def get_fnames_to_load(dataset_path, num_trajs=None):
    fnames_fpath = os.path.join(dataset_path, "shuffled_indices.txt")

    if os.path.exists(fnames_fpath):
        with open(fnames_fpath, "r") as f:
            file_suffixes = f.readlines()
            file_suffixes = [f.strip() for f in file_suffixes]

    else:
        print(f" [ utils/plan ] Could not find shuffled indices at {fnames_fpath}. Generating new shuffled indices")
        trajectories_path = os.path.join(dataset_path, "trajectories")
        plans_path = os.path.join(dataset_path, "plans")

        assert os.path.exists(plans_path), f"Plans path {plans_path} does not exist"
        assert os.path.exists(trajectories_path), f"Trajectories path {trajectories_path} does not exist"

        file_suffixes = []

        for plan_file in os.listdir(plans_path):
            if not plan_file.startswith("plan"):
                continue

            file_suffix = plan_file.split("plan_")[1]

            traj_file = f"traj_{file_suffix}"

            if not os.path.exists(os.path.join(trajectories_path, traj_file)):
                print(f" [ utils/plan ] Trajectory file {traj_file} does not exist. Skipping plan {plan_file}")
                continue

            file_suffix = file_suffix.replace(".txt", "")
            file_suffixes.append(file_suffix)

        with open(fnames_fpath, "w") as f:
            for file_suffix in file_suffixes:
                f.write(file_suffix + "\n")

    if num_trajs is not None:
        file_suffixes = file_suffixes[:num_trajs]

    return file_suffixes


def read_plans_trajectories(dataset_path, dt, file_suffix):
    plan_path = os.path.join(dataset_path, "plans", f"plan_{file_suffix}.txt")
    traj_path = os.path.join(dataset_path, "trajectories", f"traj_{file_suffix}.txt")

    orig_plan = []
    orig_trajectory = []
    total_plan_time = 0
    total_trajectory_time = 0

    with open(plan_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            line = ' '.join(line.split())
            if len(line) == 0:continue
            line = line.split(" ")
            line = [float(x) for x in line]
            orig_plan.append(line)
            total_plan_time += line[1]
    with open(traj_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            line = ' '.join(line.split())
            if len(line) == 0: continue
            line = line.split(" ")
            line = [float(x) for x in line]
            orig_trajectory.append(line)
            total_trajectory_time += dt

    plan = []
    trajectory = []

    t = 0
    i = 0
    
    ctrl_idx = 0
    ctrl_end_t = orig_plan[ctrl_idx][1]

    while i < len(orig_trajectory[:-1]):
        plan.append(orig_plan[ctrl_idx][:-1])
        trajectory.append(orig_trajectory[i])

        t += dt
        i += 1

        if t - ctrl_end_t > (dt/2):
            ctrl_idx += 1
            
            if ctrl_idx == len(orig_plan):
                break
            
            ctrl_end_t += orig_plan[ctrl_idx][1]

    trajectory.append(orig_trajectory[i])

    assert len(plan) + 1 == len(trajectory), f"Plan length {len(plan)} is not equal to trajectory length {len(trajectory)} + 1"

    plan = np.array(plan, dtype=np.float64)
    trajectory = np.array(trajectory, dtype=np.float64)
    
    return plan, trajectory

def load_plans(dataset, dataset_size=None, parallel=True, fnames=None, dt=0.002):
    """
    Load plans and trajectories from dataset.
    Trajectory length is 1 + plan length.
    """
    print(f"[ utils/plan ] Loading plans for dataset {dataset}")

    dataset_path = os.path.join("data_trajectories", dataset)
    file_suffixes = get_fnames_to_load(dataset_path, dataset_size)

    if fnames is None:
        fnames = file_suffixes

    plans = []
    trajectories = []

    if not parallel:
        for file_suffix in tqdm(fnames):
            plan, trajectory = read_plans_trajectories(dataset_path, dt, file_suffix)
            if plan is None:
                continue
            plans.append(plan)
            trajectories.append(trajectory)

    else:
        import multiprocessing as mp

        with mp.Pool(cpu_count()) as pool:
            plans, trajectories = zip(*list(tqdm(
                pool.imap(
                    partial(read_plans_trajectories, dataset_path, dt),
                    fnames
                ),
                total=len(fnames)
            )))

    assert len(plans) == len(trajectories), f"Number of plans {len(plans)} is not equal to number of trajectories {len(trajectories)}"

    return {
        "plans": plans,
        "trajectories": trajectories
    }
    
def apply_preprocess_fns(data, trajectory_preprocess_fns, plan_preprocess_fns, **preprocess_kwargs):
    for trajectory_preprocess_fn in trajectory_preprocess_fns:
        data = trajectory_preprocess_fn(data, parallel=True, **preprocess_kwargs)
    for plan_preprocess_fn in plan_preprocess_fns:
        data["plans"] = plan_preprocess_fn(data["plans"], parallel=True, **preprocess_kwargs)

    assert len(data["trajectories"]) == len(data["plans"]), f"Number of trajectories {len(data['trajectories'])} is not equal to number of plans {len(data['plans'])}"
        
    return data


def combine_plans_trajectories(plans, trajectories):
    combined = []
    for plan, trajectory in zip(plans, trajectories):
        plan = np.concatenate([np.zeros((1, plan.shape[1])), plan], axis=0)
        combined.append(np.concatenate([plan, trajectory], axis=1))
    return combined
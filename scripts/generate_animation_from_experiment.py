import argparse
import os
import re
import numpy as np
from genMoPlan.adaptive_training.animation_generator import AnimationGenerator
from genMoPlan.utils.json_args import JSONArgs
from genMoPlan import utils

def load_start_point(fname, dataset_path, observation_dim):
    """Helper to load a single trajectory's start point."""
    fpath = os.path.join(dataset_path, "trajectories", fname)
    trajectory = utils.read_trajectory(fpath, observation_dim)
    return trajectory[0]

def load_all_start_points(exp_args):
    """Loads all start points from the dataset specified in experiment args."""
    dataset_path = os.path.join(utils.get_data_trajectories_path(), exp_args.dataset)
    trajectories_path = os.path.join(dataset_path, "trajectories")

    fnames = utils.get_fnames_to_load(dataset_path, trajectories_path)
    args_list = [(fname, dataset_path, exp_args.observation_dim) for fname in fnames]

    start_points = utils.parallelize_toggle(load_start_point, args_list, parallel=True, desc="Loading all start points")
    
    return np.array(start_points)

def find_iteration_dirs(exp_path):
    """Finds and sorts iteration directories."""
    iteration_dirs = []
    for item in os.listdir(exp_path):
        if os.path.isdir(os.path.join(exp_path, item)) and item.startswith("iteration_"):
            iteration_dirs.append(item)
    
    # Sort numerically by iteration number
    iteration_dirs.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)))
    
    return [os.path.join(exp_path, d) for d in iteration_dirs]

def main():
    parser = argparse.ArgumentParser(description="Generate animations from an existing adaptive training experiment.")

    parser.add_argument(
        "--exp_path", 
        type=str, 
        required=True, 
        help="Path to the adaptive training experiment directory."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Name of the dataset used for the experiment."
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=1, 
        help="Frames per second for the animation."
    )
    args = parser.parse_args()

    # Load experiment arguments from args.json
    args_json_path = os.path.join(args.exp_path, "args.json")
    if not os.path.exists(args_json_path):
        print(f"Error: args.json not found in {args.exp_path}")
        return
    exp_args = JSONArgs(args_json_path)

    # Instantiate uncertainty class to get its name
    try:
        uncertainty_class = utils.import_class(exp_args.uncertainty)
        uncertainty_kwargs = exp_args.uncertainty_kwargs or {}
        uncertainty_obj = uncertainty_class(**uncertainty_kwargs)
        uncertainty_name = uncertainty_obj.name
    except Exception as e:
        print(f"Error instantiating uncertainty class '{exp_args.uncertainty}': {e}")
        return

    # Instantiate AnimationGenerator
    animator = AnimationGenerator(
        dataset_name=args.dataset,
        uncertainty_name=uncertainty_name
    )
    animator.animation_fps = args.fps

    # Load all dataset start points (for plotting context)
    try:
        dataset_start_points = load_all_start_points(exp_args)
    except Exception as e:
        print(f"Error loading start points: {e}")
        return

    iteration_dirs = find_iteration_dirs(args.exp_path)
    if not iteration_dirs:
        print(f"No 'iteration_*' directories found in {args.exp_path}. Nothing to do.")
        return
        
    print(f"Found {len(iteration_dirs)} iteration directories. Generating animations...")

    for it_dir in iteration_dirs:
        iteration_num_match = re.search(r'iteration_(\d+)', it_dir)
        if not iteration_num_match:
            continue
        iteration = int(iteration_num_match.group(1))

        uncertainty_path = os.path.join(it_dir, "uncertainty.npy")
        indices_path = os.path.join(it_dir, "sampled_indices.npy")

        if not os.path.exists(uncertainty_path):
            print(f"Warning: uncertainty.npy not found in {it_dir}, skipping iteration {iteration}.")
            continue
        if not os.path.exists(indices_path):
            print(f"Warning: sampled_indices.npy not found in {it_dir}, skipping iteration {iteration}.")
            continue
            
        print(f"Processing iteration {iteration}...")

        uncertainty = np.load(uncertainty_path)
        sampled_indices = np.load(indices_path)

        animator.generate_animations(
            iteration=iteration,
            uncertainty=uncertainty,
            dataset_start_points=dataset_start_points,
            sampled_indices=sampled_indices,
            save_path=args.exp_path
        )
        
    print("\nAnimation generation complete. Processes are running in the background.")
    print(f"GIF files will be saved in: {args.exp_path}")

if __name__ == "__main__":
    main() 
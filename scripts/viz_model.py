from os import path
import argparse

from mg_diffuse.utils import ROAEstimator, load_trajectories, save_trajectories_image


def visualize_generated_trajectories(dataset, num_trajs, compare, show_traj_ends, exp_path, model_state_name, only_execute_next_step, batch_size=None):
    test_trajs = load_trajectories(dataset, num_trajs)
    start_points = test_trajs[:, 0]

    roa_estimator = ROAEstimator(dataset, model_state_name, exp_path, n_runs=1, batch_size=batch_size)
    roa_estimator.start_points = start_points
    roa_estimator.generate_trajectories(compute_labels=False, discard_trajectories=False, verbose=True, save=False)

    generated_trajs = roa_estimator.trajectories[:, 0]

    image_name = "trajectories"

    if only_execute_next_step:
        image_name += "_MPC"

    image_name += f"_{num_trajs}.png"

    image_path = path.join(exp_path, image_name)

    save_trajectories_image(
        generated_trajs,
        image_path,
        verbose=True,
        comparison_trajectories=test_trajs if compare else None,
        show_traj_ends=show_traj_ends,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model trajectories")

    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name"
    )
    parser.add_argument(
        "--num_trajs",
        type=int,
        default=1000,
        help="Number of trajectories to visualize",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Add comparison trajectories to the visualization",
    )
    parser.add_argument(
        "--show_traj_ends",
        action="store_true",
        help="Show the start and end points of the trajectories",
    )
    parser.add_argument("--exp_path", type=str, help="Experiment path")
    
    parser.add_argument(
        "--exp_paths", 
        type=str, 
        nargs="+", 
        help="Multiple experiment paths. If provided, will override --exp_path"
    )
    
    parser.add_argument(
        "--model_state_name", type=str, default="best.pt", help="Model state file name"
    )
    parser.add_argument(
        "--only_execute_next_step",
        action="store_true",
        help="Only execute the next step of the trajectory like MPC",
    )
    parser.add_argument(
        "--batch_size", type=float, help="Batch size"
    )

    args = parser.parse_args()

    if args.exp_path is None and args.exp_paths is None:
        raise ValueError("Either exp_path or exp_paths must be provided")
    
    if args.exp_paths is not None:
        for exp_path in args.exp_paths:
            visualize_generated_trajectories(
                args.dataset,
                args.num_trajs,
                args.compare,
                args.show_traj_ends,
                exp_path,
                args.model_state_name,
                args.only_execute_next_step,
                args.batch_size,
            )

    else:
        visualize_generated_trajectories(
            args.dataset,
            args.num_trajs,
            args.compare,
            args.show_traj_ends,
            args.exp_path,
            args.model_state_name,
            args.only_execute_next_step,
            args.batch_size,
        )

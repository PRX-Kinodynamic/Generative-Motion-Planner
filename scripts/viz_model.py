from os import path
import argparse

from genMoPlan.utils import ROAEstimator, load_trajectories, plot_trajectories


def visualize_generated_trajectories(
        dataset, 
        num_trajs, 
        model_paths,
        model_state_name,
        observation_dim,
        batch_size=None,
    ):
    test_trajs = load_trajectories(dataset, observation_dim, num_trajs)
    start_points = test_trajs[:, 0]

    if isinstance(model_paths, str):
        model_paths = [model_paths]

    for model_path in model_paths:
        roa_estimator = ROAEstimator(dataset, model_state_name, model_path, n_runs=1, batch_size=batch_size, verbose=True)
        roa_estimator.start_points = start_points
        roa_estimator.generate_trajectories(compute_labels=False, discard_trajectories=False, save=False)
        roa_estimator.plot_trajectories()


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
    parser.add_argument("--model_path", type=str, help="Experiment path")
    
    parser.add_argument(
        "--model_paths", 
        type=str, 
        nargs="+", 
        help="Multiple experiment paths. If provided, will override --model_path"
    )

    parser.add_argument(
        "--observation_dim", 
        type=int, 
        required=True,
        help="Observation dimension"
    )
    
    parser.add_argument(
        "--model_state_name", type=str, default="best.pt", help="Model state file name"
    )

    parser.add_argument(
        "--batch_size", type=float, help="Batch size"
    )

    args = parser.parse_args()

    if args.model_path is None and args.model_paths is None:
        raise ValueError("Either model_path or model_paths must be provided")
    
    visualize_generated_trajectories(
        dataset=args.dataset,
        num_trajs=args.num_trajs,
        model_paths=args.model_paths if args.model_paths is not None else args.model_path,
        model_state_name=args.model_state_name,
        batch_size=args.batch_size,
        observation_dim=args.observation_dim,
    )

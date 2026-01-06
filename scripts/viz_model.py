from os import path
import argparse
import importlib
import numpy as np

from genMoPlan.utils import load_trajectories, get_dataset_config
from genMoPlan.eval.roa import Classifier


def visualize_generated_trajectories(
    dataset,
    num_trajs,
    model_paths,
    model_state_name,
    batch_size=None,
):
    config = get_dataset_config(dataset)
    test_trajs = load_trajectories(dataset, config["read_trajectory_fn"], num_trajs)
    start_states = np.array([traj[0] for traj in test_trajs])

    if isinstance(model_paths, str):
        model_paths = [model_paths]

    # Build the latest system definition from the dataset config.
    config_module = importlib.import_module(f"config.{dataset}")
    get_system = getattr(config_module, "get_system", None)
    if get_system is None:
        raise ValueError(
            f"Config module for dataset '{dataset}' does not define get_system()."
        )
    system = get_system()

    for model_path in model_paths:
        classifier = Classifier(
            dataset=dataset,
            model_state_name=model_state_name,
            model_path=model_path,
            n_runs=1,
            batch_size=batch_size,
            verbose=True,
            system=system,
        )
        classifier.start_states = start_states
        classifier.generate_trajectories(discard_trajectories=False, save=False)
        classifier.plot_trajectories()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model trajectories")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--num_trajs",
        type=int,
        default=1000,
        help="Number of trajectories to visualize",
    )

    parser.add_argument("--model_path", type=str, help="Experiment path")

    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        help="Multiple experiment paths. If provided, will override --model_path",
    )

    parser.add_argument(
        "--model_state_name", type=str, default="best.pt", help="Model state file name"
    )

    parser.add_argument("--batch_size", type=float, help="Batch size")

    args = parser.parse_args()

    if args.model_path is None and args.model_paths is None:
        raise ValueError("Either model_path or model_paths must be provided")

    visualize_generated_trajectories(
        dataset=args.dataset,
        num_trajs=args.num_trajs,
        model_paths=(
            args.model_paths if args.model_paths is not None else args.model_path
        ),
        model_state_name=args.model_state_name,
        batch_size=args.batch_size,
    )

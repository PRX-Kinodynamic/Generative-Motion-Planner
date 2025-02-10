from os import path
import argparse

import mg_diffuse.utils.model
from mg_diffuse import utils


def main(args):
    exp_path = path.join("experiments", args.dataset, args.path_prefix, args.exp_name)

    test_trajs = utils.load_test_trajectories(args.dataset, args.num_trajs)
    start_points = test_trajs[:, 0]

    model, model_args = mg_diffuse.utils.model.load_model(exp_path, args.model_state_name)

    generated_trajs = utils.generate_trajectories(
        model, model_args, start_points, args.only_execute_next_step, verbose=True
    )

    image_name = "trajectories"

    if args.only_execute_next_step:
        image_name += "_MPC"

    image_name += f"_{args.num_trajs}.png"

    image_path = path.join(exp_path, image_name)

    utils.save_trajectories_image(
        generated_trajs,
        image_path,
        verbose=True,
        comparison_trajectories=test_trajs if args.compare else None,
        show_traj_ends=args.show_traj_ends,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model trajectories")

    parser.add_argument(
        "--path_prefix",
        type=str,
        default="diffusion",
        help="Path prefix for experiments",
    )
    parser.add_argument(
        "--dataset", type=str, default="pendulum_lqr_5k", help="Dataset name"
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
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--model_state_name", type=str, required=True, help="Model state file name"
    )
    parser.add_argument(
        "--only_execute_next_step",
        action="store_true",
        help="Only execute the next step of the trajectory like MPC",
    )

    main(parser.parse_args())

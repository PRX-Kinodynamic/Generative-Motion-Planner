from os import path
from argparse import ArgumentParser

import mg_diffuse.utils as utils


def main(args):
    test_trajectories = utils.load_test_trajectories(args.dataset, args.num_trajs)
    test_image_path = path.join(
        "data_trajectories", args.dataset, f"test_trajectories_{args.num_trajs}.png"
    )

    utils.save_trajectories_image(test_trajectories, test_image_path, verbose=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize dataset trajectories")
    parser.add_argument(
        "--num_trajs", type=int, default=100, help="Number of trajectories to visualize"
    )
    parser.add_argument(
        "--path_prefix",
        type=str,
        default="diffusion",
        help="Path prefix for experiments",
    )
    parser.add_argument(
        "--dataset", type=str, default="pendulum_lqr_5k", help="Dataset name"
    )

    main(parser.parse_args())

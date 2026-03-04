from os import path
from argparse import ArgumentParser

import genMoPlan.utils as utils


def main(args):
    config = utils.get_dataset_config(args.dataset)
    trajectories = utils.load_trajectories(args.dataset, config["read_trajectory_fn"], args.num_trajs, parallel=not args.no_parallel)

    fname = f'trajectories_{args.num_trajs}'

    img_path = path.join(
        utils.get_data_trajectories_path(), args.dataset, fname + ".png"
    )

    utils.plot_trajectories(trajectories, img_path, verbose=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize dataset trajectories")
    parser.add_argument(
        "--num_trajs", type=int, default=100, help="Number of trajectories to visualize"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name"
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Do not use parallel processing",
    )

    main(parser.parse_args())

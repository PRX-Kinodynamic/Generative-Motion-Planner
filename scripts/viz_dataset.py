from os import path
from argparse import ArgumentParser

import genMoPlan.utils as utils


def main(args):
    trajectories = utils.load_trajectories(args.dataset, args.num_trajs, parallel=not args.no_parallel)

    if args.apply_config:
        import importlib

        config_module = importlib.import_module(f"config.{args.dataset}")
        config = config_module.base["diffusion"]

        if args.variation is not None:
            variation_config = getattr(config_module, args.variation)["diffusion"]
            config.update(variation_config)
            fname = f"trajectories_{args.num_trajs}_{args.dataset}_{args.variation}"
        else:
            fname = f"trajectories_{args.num_trajs}_{args.dataset}"

        preprocess_fns = config["preprocess_fns"]
        preprocess_kwargs = config["preprocess_kwargs"]

        for preprocess_fn in preprocess_fns:
            trajectories = preprocess_fn(trajectories, **preprocess_kwargs, parallel=not args.no_parallel)


    else:
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
        "--path_prefix",
        type=str,
        default="diffusion",
        help="Path prefix for experiments",
    )
    parser.add_argument(
        "--show_traj_ends",
        action="store_true",
        help="Show the start and end points of the trajectories",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name"
    )

    parser.add_argument(
        "--apply_config",
        action="store_true",
        help="Apply configuration settings from config file",
    )

    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Do not use parallel processing",
    )

    parser.add_argument(
        "--variation",
        type=str,
        default=None,
        help="Variation of the dataset to use",
    )

    main(parser.parse_args())

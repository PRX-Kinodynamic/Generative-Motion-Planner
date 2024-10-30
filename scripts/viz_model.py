from os import path
import argparse

from mg_diffuse import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize trajectories')

    parser.add_argument('--path_prefix', type=str, default='diffusion', help='Path prefix for experiments')
    parser.add_argument('--dataset', type=str, default='pendulum_lqr_5k', help='Dataset name')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--model_state_name', type=str, required=True, help='Model state file name')
    parser.add_argument('--only_execute_next_step', action='store_true',
                        help='Only execute the next step of the trajectory like MPC')

    args = parser.parse_args()

    utils.visualize_trajectories(
        path.join('experiments', args.dataset, args.path_prefix, args.exp_name),
        args.model_state_name,
        args.only_execute_next_step,
        sampling_limits=args.sampling_limits,
        granularity=args.granularity,
    )
from os import path
from argparse import ArgumentParser

import numpy as np

import mg_diffuse.utils as utils
from mg_diffuse.datasets import load_trajectories

def main():
    parser = ArgumentParser(description='Visualize trajectories')
    parser.add_argument('--num', type=int, default=100, help='Number of trajectories to visualize')

    args = parser.parse_args()

    dataset = "pendulum_lqr_5k"

    trajectories = load_trajectories(dataset)

    image_path = path.join('data_trajectories', dataset, 'trajectories_image.png')

    indices = np.random.choice(len(trajectories), args.num, replace=False)

    utils.save_trajectories_image(trajectories[indices], image_path, verbose=True)

if __name__ == '__main__':
    main()
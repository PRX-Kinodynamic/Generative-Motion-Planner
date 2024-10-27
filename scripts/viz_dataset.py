from os import path

import mg_diffuse.utils as utils
from mg_diffuse.datasets import load_trajectories

def main():
    dataset = "pendulum_lqr_5k"

    trajectories = load_trajectories(dataset)

    image_path = path.join('data_trajectories', dataset, 'trajectories_image.png')

    utils.save_trajectories_image(trajectories, image_path, verbose=True, parallel=True)

if __name__ == '__main__':
    main()
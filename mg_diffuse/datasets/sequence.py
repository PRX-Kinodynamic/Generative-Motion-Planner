from collections import namedtuple
from os import cpu_count

import numpy as np
import torch
import os

from tqdm import tqdm

from .normalization import  *

Batch = namedtuple('Batch', 'trajectories conditions')


def read_trajectory(sequence_path):
    with open(sequence_path, 'r') as f:
        lines = f.readlines()

    trajectory = []

    for line in lines:
        state = line.split(',')
        state = [float(s) for s in state]

        trajectory.append(state)

    return trajectory

def load_trajectories(dataset, parallel=True):
    """
        load dataset from directory
    """

    trajectories_path = os.path.join('data_trajectories', dataset, 'trajectories')

    try:
        fnames = os.listdir(trajectories_path)
    except FileNotFoundError:
        raise ValueError(f'Could not find dataset at {trajectories_path}')

    trajectories = []

    print(f'[ datasets/sequence ] Loading trajectories from {trajectories_path}')

    fpaths = [os.path.join(trajectories_path, fname) for fname in fnames]

    if not parallel:
        for fpath in tqdm(fpaths):
            if not fpath.endswith('.txt'):
                continue
            trajectories.append(read_trajectory(fpath))
    else:
        import multiprocessing as mp

        # read trajectories in parallel with tqdm progress bar
        with mp.Pool(cpu_count()) as pool:
            trajectories = list(tqdm(pool.imap(read_trajectory, fpaths), total=len(fpaths)))

    return np.array(trajectories, dtype=np.float32)


class TrajectoryDataset(torch.utils.data.Dataset):
    normed_trajectories = None

    def __init__(self, dataset=None, horizon=64,
        normalizer='GaussianNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):

        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        if dataset is None:
            raise ValueError('dataset not specified')

        trajectories = load_trajectories(dataset)

        path_lengths = [len(trajectory) for trajectory in trajectories]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizer = normalizer(trajectories)
        self.indices = self.make_indices(path_lengths, horizon)

        self.observation_dim = trajectories.shape[-1]
        self.trajectories = trajectories
        self.n_episodes = trajectories.shape[0]
        self.path_lengths = path_lengths

        self.normalize()

    def normalize(self):
        """
            normalize fields that will be predicted by the diffusion model
        """
        array = self.trajectories.reshape(self.n_episodes*self.max_path_length, -1)
        normed = self.normalizer(array)
        self.normed_trajectories = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        """
            makes indices for sampling from dataset;
            each index maps to a datapoint
        """
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        trajectories = self.normed_trajectories[path_ind, start:end]

        conditions = self.get_conditions(trajectories)
        batch = Batch(trajectories, conditions)
        return batch
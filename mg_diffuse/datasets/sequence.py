from collections import namedtuple

import torch

from .normalization import *
from flow_matching.utils.manifolds import Euclidean


Batch = namedtuple("Batch", "trajectories conditions")

class TrajectoryDataset(torch.utils.data.Dataset):
    normed_trajectories = None

    def __init__(
        self,
        dataset=None,
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=(),
        preprocess_kwargs={},
        dataset_size=None,
        max_path_length=1000,
        use_padding=True,
        manifold=False
    ):
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        if dataset is None:
            raise ValueError("dataset not specified")

        from mg_diffuse.utils.trajectory import load_trajectories

        trajectories = load_trajectories(dataset, dataset_size)

        for preprocess_fn in preprocess_fns:
            trajectories = preprocess_fn(trajectories, **preprocess_kwargs)

        path_lengths = [len(trajectory) for trajectory in trajectories]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        if manifold:
            self.normalizer = normalizer(manifold, trajectories)
        else:
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
        array = self.trajectories.reshape(self.n_episodes * self.max_path_length, -1)
        normed = self.normalizer(array)
        self.normed_trajectories = normed.reshape(
            self.n_episodes, self.max_path_length, -1
        )

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
            for start in range(max_start+1):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        trajectories = self.normed_trajectories[path_ind, start:end]



        conditions = self.get_conditions(trajectories)
        batch = Batch(trajectories, conditions)
        return batch

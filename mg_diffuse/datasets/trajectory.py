from collections import namedtuple

import torch

from .normalization import *


Batch = namedtuple("Batch", "trajectories conditions")

class TrajectoryDataset(torch.utils.data.Dataset):
    normed_trajectories = None

    def __init__(
        self,
        dataset=None,
        horizon=64,
        stride=1,
        normalizer="LimitsNormalizer",
        normalizer_params={},
        preprocess_fns=(),
        preprocess_kwargs={},
        dataset_size=None,
        max_path_length=1000,
        use_padding=True,
        use_plans=False,
    ):
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.stride = stride

        if dataset is None:
            raise ValueError("dataset not specified")
        
        if use_plans:
            from mg_diffuse.utils.plan import load_plans, combine_plans_trajectories
            data = load_plans(dataset, dataset_size)
            trajectories = data["trajectories"]
            plans = data["plans"]
            trajectories = combine_plans_trajectories(plans, trajectories)
        else:
            from mg_diffuse.utils.trajectory import load_trajectories

            trajectories = load_trajectories(dataset, dataset_size)

        for preprocess_fn in preprocess_fns:
            trajectories = preprocess_fn(trajectories, **preprocess_kwargs)

        trajectories = np.array(trajectories, dtype=np.float32)

        path_lengths = [len(trajectory) for trajectory in trajectories]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizer = normalizer(trajectories, **normalizer_params)
        self.indices = self.make_indices(path_lengths)

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

    def make_indices(self, path_lengths):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        indices = []
        actual_horizon = 1 + (self.horizon - 1) * self.stride
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 3, self.max_path_length - 3)
            if not self.use_padding:
                max_start = min(max_start, path_length - actual_horizon)
            for start in range(max_start+1):
                end = start + actual_horizon
                indices.append((i, start, end))
                breakpoint()
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

        trajectory = self.normed_trajectories[path_ind, start:end+1:self.stride]

        if self.use_history_padding and len(trajectory) < self.horizon:
            padding_length = self.horizon - len(trajectory)
            padding = np.tile(trajectory[-1], (padding_length, 1))
            trajectory = np.concatenate([trajectory, padding], axis=0)

        assert len(trajectory) == self.horizon, "Trajectory length is not as expected"

        conditions = self.get_conditions(trajectory)
        batch = Batch(trajectories=trajectory, conditions=conditions)
        return batch

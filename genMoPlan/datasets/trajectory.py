from collections import namedtuple

import torch
import numpy as np

from genMoPlan.datasets.normalization import *
from genMoPlan.datasets.utils import apply_padding, make_indices
from genMoPlan.utils import load_trajectories


Batch = namedtuple("Batch", "trajectory conditions query")

class TrajectoryDataset(torch.utils.data.Dataset):
    normed_trajectories = None

    def __init__(
        self,
        dataset: str = None,
        horizon_length: int = 31,
        history_length: int = 1,
        stride: int = 1,
        observation_dim: int = None,
        trajectory_normalizer: str = "LimitsNormalizer",
        normalizer_params: dict = {},
        trajectory_preprocess_fns: tuple = (),
        preprocess_kwargs: dict = {},
        dataset_size: int = None,
        use_horizon_padding: bool = False,
        use_history_padding: bool = False,
        is_history_conditioned: bool = True, # Otherwise it is provided as a query that is not predicted by the model
        is_validation: bool = False,
        **kwargs,
    ):
        self.horizon_length = horizon_length
        self.history_length = history_length
        self.stride = stride
        self.use_horizon_padding = use_horizon_padding
        self.use_history_padding = use_history_padding
        self.is_history_conditioned = is_history_conditioned
        self.observation_dim = observation_dim

        trajectories = self._load_data(
            dataset,
            dataset_size, 
            trajectory_preprocess_fns, 
            preprocess_kwargs,
            is_validation,
        )

        traj_lengths = [len(trajectory) for trajectory in trajectories]
        
        self.indices = make_indices(
            traj_lengths, self.history_length, self.use_history_padding, self.horizon_length, self.use_horizon_padding, self.stride
        )

        self.n_episodes = len(trajectories)
        self.traj_lengths = traj_lengths

        self.normed_trajectories = self._normalize(
            trajectories, 
            trajectory_normalizer, 
            normalizer_params
        )

    def _load_data(self, dataset, dataset_size, trajectory_preprocess_fns, preprocess_kwargs, is_validation):
        trajectories = load_trajectories(dataset, self.observation_dim, dataset_size, load_reverse=is_validation)
        for trajectory_preprocess_fn in trajectory_preprocess_fns:
            trajectories = trajectory_preprocess_fn(trajectories, **preprocess_kwargs["trajectory"])
        return trajectories

    def _normalize(self, trajectories, trajectory_normalizer=None, normalizer_params=None):
        """
        normalize fields that will be predicted

        First, aggregate all trajectories into a single array
        Then, normalize the aggregated array
        Then, split the normalized array back into individual trajectories. 
        """
        if trajectory_normalizer is None:
            return [
                torch.FloatTensor(trajectory)
                for trajectory in trajectories
            ]
        
        print(f"[ datasets/trajectory ] Normalizing trajectories")

        all_trajectories = np.concatenate(trajectories, axis=0)

        if type(trajectory_normalizer) == str:
            trajectory_normalizer = eval(trajectory_normalizer)

        trajectory_normalizer = trajectory_normalizer(X=trajectories, params=normalizer_params["trajectory"])
        normed_all_trajectories = trajectory_normalizer(all_trajectories)

        # Split all trajectories into individual trajectories
        normed_trajectories = []
        traj_start_idx = 0
        traj_lengths = [len(traj) for traj in trajectories]
        
        for traj_length in traj_lengths:
            traj_end_idx = traj_start_idx + traj_length
            normed_traj = normed_all_trajectories[traj_start_idx:traj_end_idx]
            
            normed_trajectories.append(torch.FloatTensor(normed_traj, device='cpu'))
            traj_start_idx = traj_end_idx
        
        return normed_trajectories

    def get_conditions(self, history):
        """
        conditions on current observation for planning
        """
        return dict(enumerate(history))

    def get_query(self, history):
        """
        query on current observation for planning
        """
        if self.is_history_conditioned:
            return torch.zeros(0, dtype=torch.float32, device='cpu')
        else:
            return history
        
    def get_trajectory(self, history, horizon):
        """
        trajectory on current observation for planning
        """
        if self.is_history_conditioned:
            return torch.cat([history, horizon], dim=0)
        else:
            return horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, history_start, history_end, horizon_start, horizon_end = self.indices[idx]

        # Extract the actual available data
        trajectory_data = self.normed_trajectories[path_ind]
        
        # Get actual data for history and horizon
        history = trajectory_data[history_start:history_end:self.stride]
        horizon = trajectory_data[horizon_start:horizon_end:self.stride]

        assert len(history) > 0, "History is empty"

        if self.use_history_padding:
            history = apply_padding(history, self.history_length, pad_left=True)

        if self.use_horizon_padding:
            if len(horizon) > 0:
                pad_value = horizon[-1]
            else:
                pad_value = history[-1]

            horizon =apply_padding(horizon, self.horizon_length, pad_left=False, pad_value=pad_value)

        assert len(history) == self.history_length, f"History length is {len(history)}, expected {self.history_length}"
        assert len(horizon) == self.horizon_length, f"Horizon length is {len(horizon)}, expected {self.horizon_length}"
        

        trajectory = self.get_trajectory(history, horizon)
        conditions = self.get_conditions(history)
        query = self.get_query(history)

        batch = Batch(trajectory=trajectory, conditions=conditions, query=query)

        return batch

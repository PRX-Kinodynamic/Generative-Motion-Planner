from collections import namedtuple

import torch
from tqdm import tqdm

from .normalization import *


Batch = namedtuple("Batch", "trajectories conditions")

class TrajectoryDataset(torch.utils.data.Dataset):
    normed_trajectories = None

    def __init__(
        self,
        dataset=None,
        horizon_length=64,
        horizon_stride=1,
        history_stride=1,
        history_length=1,
        trajectory_normalizer="LimitsNormalizer",
        normalizer_params={},
        trajectory_preprocess_fns=(),
        preprocess_kwargs={},
        dataset_size=None,
        use_horizon_padding=True,
        use_history_padding=True,
        use_plan=False,
        dt=0.002,
    ):
        self.horizon_length = horizon_length
        self.horizon_stride = horizon_stride
        self.history_stride = history_stride
        self.history_length = history_length
        self.use_horizon_padding = use_horizon_padding
        self.use_history_padding = use_history_padding

        if dataset is None:
            raise ValueError("dataset not specified")
        
        from mg_diffuse.utils.plan import load_plans, combine_plans_trajectories
        data = load_plans(dataset, dataset_size, dt=dt)
        trajectories = data["trajectories"]
        
        if use_plan:
            plans = data["plans"]
            trajectories = combine_plans_trajectories(plans, trajectories)
        else:
            pass
            # from mg_diffuse.utils.trajectory import load_trajectories

            # trajectories = load_trajectories(dataset, dataset_size)

        for trajectory_preprocess_fn in trajectory_preprocess_fns:
            trajectories = trajectory_preprocess_fn(trajectories, **preprocess_kwargs)

        path_lengths = [len(trajectory) for trajectory in trajectories]

        if type(trajectory_normalizer) == str:
            trajectory_normalizer = eval(trajectory_normalizer)

        self.trajectory_normalizer = trajectory_normalizer(trajectories, params=normalizer_params["trajectory"])
        self.indices = self.make_indices(path_lengths)

        self.observation_dim = len(trajectories[0][0])
        self.trajectories = trajectories
        self.n_episodes = len(trajectories)
        self.path_lengths = path_lengths

        self.normalize()

    def normalize(self):
        """
        normalize fields that will be predicted by the diffusion model
        """
        normed_trajectories = []

        # Process trajectories and plans in a single loop
        print(f"[ datasets/trajectory ] Normalizing trajectories")
        for i in tqdm(range(len(self.trajectories))):
            # Normalize trajectory
            normed_traj = self.trajectory_normalizer(self.trajectories[i])
            normed_traj_tensor = torch.FloatTensor(normed_traj)
            normed_trajectories.append(normed_traj_tensor)
           
        self.normed_trajectories = normed_trajectories

    def make_indices(self, path_lengths):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        indices = []
        
        actual_horizon_length = self.horizon_length * self.horizon_stride
        actual_history_length = self.history_length * self.history_stride

        for i, path_length in enumerate(path_lengths):
            max_start = path_length - (actual_history_length + actual_horizon_length)
            
            if self.use_horizon_padding:
                raise NotImplementedError("Horizon padding not implemented")
            
            if self.use_history_padding:
                raise NotImplementedError("History padding not implemented")
            
            for history_start in range(max_start+1):
                history_end = history_start + actual_history_length
                horizon_start = history_end
                horizon_end = horizon_start + actual_horizon_length
                indices.append((i, history_start, history_end, horizon_start, horizon_end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, history):
        """
        condition on current observation for planning
        """
        return dict(enumerate(history))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, history_start, history_end, horizon_start, horizon_end = self.indices[idx]

        history = self.normed_trajectories[path_ind][history_start:history_end:self.history_stride]
        horizon = self.normed_trajectories[path_ind][horizon_start:horizon_end:self.horizon_stride]
        trajectory = np.concatenate([history, horizon], axis=0)

        if self.use_history_padding:
            raise NotImplementedError("History padding not implemented")

        if self.use_horizon_padding:
            raise NotImplementedError("Horizon padding not implemented")

        assert len(history) == self.history_length, "History length is not as expected"
        assert len(horizon) == self.horizon_length, "Horizon length is not as expected"

        conditions = self.get_conditions(history)
        batch = Batch(trajectories=trajectory, conditions=conditions)
        return batch

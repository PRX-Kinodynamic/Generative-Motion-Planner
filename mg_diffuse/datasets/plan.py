from collections import namedtuple

import torch
import numpy as np
from tqdm import tqdm

from mg_diffuse.utils.plan import load_plans, apply_preprocess_fns
from .normalization import *


Batch = namedtuple("Batch", "trajectories conditions")

class PlanDataset(torch.utils.data.Dataset):
    normed_trajectories = None
    normed_plans = None

    def __init__(
        self,
        dataset=None,
        horizon=50,
        history_length=50,
        trajectory_normalizer="LimitsNormalizer",
        plan_normalizer="LimitsNormalizer",
        trajectory_preprocess_fns=(),
        plan_preprocess_fns=(),
        preprocess_kwargs={},
        dataset_size=None,
        dt=0.002,
        normalizer_params={},
        use_history_padding=False,
        **kwargs,
    ):
        self.horizon = horizon
        self.history_length = history_length
        self.use_history_padding = use_history_padding
        if dataset is None:
            raise ValueError("dataset not specified")

        data = load_plans(dataset, dataset_size, dt=dt)

        data = apply_preprocess_fns(data, trajectory_preprocess_fns, plan_preprocess_fns, **preprocess_kwargs)

        trajectories = data["trajectories"]
        plans = data["plans"]

        traj_lengths = [len(trajectory) for trajectory in trajectories]

        if type(trajectory_normalizer) == str:
            trajectory_normalizer = eval(trajectory_normalizer)
        if type(plan_normalizer) == str:
            plan_normalizer = eval(plan_normalizer)

        self.trajectory_normalizer = trajectory_normalizer(trajectories, params=normalizer_params["trajectory"])
        self.plan_normalizer = plan_normalizer(plans, params=normalizer_params["plan"])

        self.indices = self.make_indices(traj_lengths, horizon, history_length)

        self.observation_dim = len(trajectories[0][0])
        self.plan_dim = len(plans[0][0])
        self.trajectories = trajectories
        self.plans = plans
        self.n_episodes = len(trajectories)
        self.traj_lengths = traj_lengths

        self.normalize()

        print(f"[ datasets/plan ] Dataset size: {len(self.indices)}")

    def normalize(self):
        """
        normalize fields that will be predicted by the diffusion model
        """
        normed_trajectories = []
        normed_plans = []
        
        # Process trajectories and plans in a single loop
        print(f"[ datasets/plan ] Normalizing trajectories and plans")
        for i in tqdm(range(len(self.trajectories))):
            # Normalize trajectory
            normed_traj = self.trajectory_normalizer(self.trajectories[i])
            normed_traj_tensor = torch.FloatTensor(normed_traj)
            normed_trajectories.append(normed_traj_tensor)
            
            # Normalize plan
            normed_plan = self.plan_normalizer(self.plans[i])
            normed_plan_tensor = torch.FloatTensor(normed_plan)
            normed_plans.append(normed_plan_tensor)
        
        self.normed_trajectories = normed_trajectories
        self.normed_plans = normed_plans

    def make_indices(self, traj_lengths, horizon_length, history_length):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        print(f"[ datasets/plan ] Preparing indices for dataset")

        indices = []

        for i, traj_length in tqdm(enumerate(traj_lengths)):
            plan_length = traj_length - 1
            max_horizon_start_idx = plan_length - horizon_length  # +1 because we want to include this index
            
            for horizon_start_idx in range(1, max_horizon_start_idx + 1):
                # Calculate horizon indices
                horizon_end_idx = horizon_start_idx + horizon_length
                
                # Calculate history indices
                history_end_idx = horizon_start_idx  # History ends right before horizon starts
                # Use max history length possible, but minimum of 1
                history_start_idx = max(0, history_end_idx - history_length)
                
                # Store both horizon and history indices
                indices.append((i, history_start_idx, history_end_idx, horizon_start_idx, horizon_end_idx))

        indices = np.array(indices)

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, history_start, history_end, horizon_start, horizon_end = self.indices[idx]

        # Get both history and horizon trajectories
        history = self.normed_trajectories[path_ind][history_start:history_end]
        horizon = self.normed_plans[path_ind][horizon_start:horizon_end]

        # Pad history if needed
        if self.use_history_padding:
            current_history_length = history_end - history_start
            if current_history_length < self.history_length:
                padding_length = self.history_length - current_history_length
                padding = history[0].unsqueeze(0).repeat(padding_length, 1)
                history = torch.cat([padding, history], dim=0)


        # Create batch with proper field names: trajectories and conditions
        batch = Batch(trajectories=horizon, conditions=history)
        return batch

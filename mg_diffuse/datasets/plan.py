from collections import namedtuple
import math

import torch
import numpy as np

from mg_diffuse.utils.plan import load_plans, apply_preprocess_fns
from mg_diffuse.datasets.normalization import *
from mg_diffuse.datasets.utils import apply_padding, make_indices

Batch = namedtuple("Batch", "plan query")

class PlanDataset(torch.utils.data.Dataset):
    normed_trajectories = None
    normed_plans = None

    def __init__(
        self,
        dataset=None,
        horizon_length=31,
        history_length=1,
        stride=1,
        trajectory_normalizer="LimitsNormalizer",
        plan_normalizer="LimitsNormalizer",
        trajectory_preprocess_fns=(),
        plan_preprocess_fns=(),
        preprocess_kwargs={},
        dataset_size=None,
        dt=0.002,
        normalizer_params={},
        use_history_padding=False,
        use_horizon_padding=False,
        **kwargs,
    ):
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.stride = stride
        self.use_history_padding = use_history_padding
        self.use_horizon_padding = use_horizon_padding

        trajectories, plans = self._load_data(dataset, dataset_size, dt, trajectory_preprocess_fns, plan_preprocess_fns, preprocess_kwargs)

        traj_lengths = [len(trajectory) for trajectory in trajectories]
        plan_lengths = [len(plan) for plan in plans]

        self.indices = make_indices(
            plan_lengths, 
            self.history_length, 
            self.use_history_padding, 
            self.horizon_length, 
            self.use_horizon_padding, 
            self.stride,
        )

        self.observation_dim = len(trajectories[0][0])
        self.plan_dim = len(plans[0][0])
        self.n_episodes = len(trajectories)
        self.traj_lengths = traj_lengths
        self.plan_lengths = plan_lengths

        self.normed_trajectories, self.normed_plans = self._normalize(
            trajectories, 
            plans, 
            trajectory_normalizer, 
            plan_normalizer, 
            normalizer_params
        )

    def _load_data(self, dataset, dataset_size, dt, trajectory_preprocess_fns, plan_preprocess_fns, preprocess_kwargs):
        data = load_plans(dataset, dataset_size, dt=dt)
        data = apply_preprocess_fns(data, trajectory_preprocess_fns, plan_preprocess_fns, **preprocess_kwargs)

        return data["trajectories"], data["plans"]

    def _normalize(self, trajectories, plans, trajectory_normalizer, plan_normalizer, normalizer_params):
        """
        normalize fields that will be predicted

        First, aggregate all trajectories and plans into a single array
        Then, normalize the aggregated array
        Then, split the normalized array back into individual trajectories and plans. 
        """
        print(f"[ datasets/plan ] Normalizing trajectories and plans")

        if type(trajectory_normalizer) == str:
            trajectory_normalizer = eval(trajectory_normalizer)
        trajectory_normalizer = trajectory_normalizer(X=trajectories, params=normalizer_params["trajectory"])
        traj_lengths = [len(traj) for traj in trajectories]
        all_trajectories = np.concatenate(trajectories, axis=0)
        normed_all_trajectories = trajectory_normalizer(all_trajectories)
        
        if type(plan_normalizer) == str:
            plan_normalizer = eval(plan_normalizer)
        plan_normalizer = plan_normalizer(X=plans, params=normalizer_params["plan"])
        plan_lengths = [len(plan) for plan in plans]
        all_plans = np.concatenate(plans, axis=0)
        normed_all_plans = plan_normalizer(all_plans)
        
        normed_trajectories = []
        normed_plans = []
        
        traj_start_idx = 0
        plan_start_idx = 0
        
        for i in range(len(trajectories)):
            traj_end_idx = traj_start_idx + traj_lengths[i]
            normed_traj = normed_all_trajectories[traj_start_idx:traj_end_idx]
            normed_traj_tensor = torch.FloatTensor(normed_traj)
            normed_trajectories.append(normed_traj_tensor)
            traj_start_idx = traj_end_idx
            
            plan_end_idx = plan_start_idx + plan_lengths[i]
            normed_plan = normed_all_plans[plan_start_idx:plan_end_idx]
            normed_plan_tensor = torch.FloatTensor(normed_plan)
            normed_plans.append(normed_plan_tensor)
            plan_start_idx = plan_end_idx
        
        return normed_trajectories, normed_plans

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, history_start, history_end, horizon_start, horizon_end = self.indices[idx]

        # Get both history and horizon trajectories
        # Apply stride to sample fewer points in the history
        history = self.normed_trajectories[path_ind][history_start:history_end:self.stride]
        # Apply stride to sample fewer points in the horizon
        horizon = self.normed_plans[path_ind][horizon_start:horizon_end:self.stride]

        assert len(history) > 0, "History is empty"
        assert len(horizon) > 0, "Horizon is empty"

        if self.use_history_padding:
            history = apply_padding(history, self.history_length, pad_left=True)
        if self.use_horizon_padding:
            horizon = apply_padding(horizon, self.horizon_length, pad_left=False)
        
        assert len(history) == self.history_length, f"History length is {len(history)}, expected {self.history_length}"
        assert len(horizon) == self.horizon_length, f"Horizon length is {len(horizon)}, expected {self.horizon_length}"

        batch = Batch(plan=horizon, query=history)

        return batch

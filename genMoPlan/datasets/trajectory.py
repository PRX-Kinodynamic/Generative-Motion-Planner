from collections import namedtuple

import torch
import numpy as np

from genMoPlan.datasets.normalization import *
from genMoPlan.datasets.utils import apply_padding, make_indices
from genMoPlan.utils.arrays import to_torch
from genMoPlan.utils.plan import apply_preprocess_fns, combine_plan_trajectory, load_plans


Batch = namedtuple("Batch", "trajectory conditions query")

class TrajectoryDataset(torch.utils.data.Dataset):
    normed_trajectories = None

    def __init__(
        self,
        dataset: str = None,
        horizon_length: int = 31,
        history_length: int = 1,
        stride: int = 1,
        trajectory_normalizer: str = "LimitsNormalizer",
        plan_normalizer: str = "LimitsNormalizer",
        normalizer_params: dict = {},
        trajectory_preprocess_fns: tuple = (),
        plan_preprocess_fns: tuple = (),
        preprocess_kwargs: dict = {},
        dataset_size: int = None,
        use_horizon_padding: bool = False,
        use_history_padding: bool = False,
        use_plan: bool = False,
        dt: float = None,
        is_history_conditioned: bool = True, # Otherwise it is provided as a query that is not predicted by the model
        is_validation: bool = False,
        **kwargs,
    ):
        self.horizon_length = horizon_length
        self.history_length = history_length
        self.stride = stride
        self.use_horizon_padding = use_horizon_padding
        self.use_history_padding = use_history_padding
        self.use_plan = use_plan
        self.is_history_conditioned = is_history_conditioned


        trajectories, plans = self._load_data(
            dataset,
            dataset_size, 
            dt, 
            trajectory_preprocess_fns, 
            plan_preprocess_fns, 
            preprocess_kwargs,
            is_validation,
        )

        traj_lengths = [len(trajectory) for trajectory in trajectories]
        
        self.indices = make_indices(
            traj_lengths, self.history_length, self.use_history_padding, self.horizon_length, self.use_horizon_padding, self.stride
        )

        self.observation_dim = len(trajectories[0][0]) if not self.use_plan else (len(trajectories[0][0]) + len(plans[0][0]))
        self.n_episodes = len(trajectories)
        self.traj_lengths = traj_lengths

        self.normed_trajectories = self._normalize(
            trajectories, 
            plans, 
            trajectory_normalizer, 
            plan_normalizer, 
            normalizer_params
        )

    def _load_data(self, dataset, dataset_size, dt, trajectory_preprocess_fns, plan_preprocess_fns, preprocess_kwargs, is_validation):
        if self.use_plan:
            from genMoPlan.utils.plan import apply_preprocess_fns, load_plans

            data = load_plans(dataset, dataset_size, dt=dt)
            data = apply_preprocess_fns(data, trajectory_preprocess_fns, plan_preprocess_fns, preprocess_kwargs)
            return data["trajectories"], data["plans"]
        else:
            from genMoPlan.utils.trajectory import load_trajectories

            trajectories = load_trajectories(dataset, dataset_size, load_reverse=is_validation)
            for trajectory_preprocess_fn in trajectory_preprocess_fns:
                trajectories = trajectory_preprocess_fn(trajectories, **preprocess_kwargs["trajectory"])
            return trajectories, None

    def _normalize(self, trajectories, plans=None, trajectory_normalizer=None, plan_normalizer=None, normalizer_params=None):
        """
        normalize fields that will be predicted

        First, aggregate all trajectories and plans into a single array
        Then, normalize the aggregated array
        Then, split the normalized array back into individual trajectories and plans. 
        If use_plan is True, then combine the plans and trajectories into a single trajectory.
        """
        all_trajectories = np.concatenate(trajectories, axis=0)

        if trajectory_normalizer is not None:
            print(f"[ datasets/trajectory ] Normalizing trajectories")
            if type(trajectory_normalizer) == str:
                trajectory_normalizer = eval(trajectory_normalizer)
            trajectory_normalizer = trajectory_normalizer(X=trajectories, params=normalizer_params["trajectory"])
            normed_all_trajectories = trajectory_normalizer(all_trajectories)
        else:
            normed_all_trajectories = all_trajectories

        if self.use_plan:
            assert plans is not None, "Plans are required when use_plan is True"

            print(f"[ datasets/trajectory ] Normalizing plans")

            plan_lengths = [len(plan) for plan in plans]
            all_plans = np.concatenate(plans, axis=0)

            if plan_normalizer is not None:
                if type(plan_normalizer) == str:
                    plan_normalizer = eval(plan_normalizer)
                plan_normalizer = plan_normalizer(X=plans, params=normalizer_params["plan"])
                normed_all_plans = plan_normalizer(all_plans)
            else:
                normed_all_plans = all_plans

        # Split all trajectories and plans into individual trajectories and plans
        normed_trajectories = []
        traj_start_idx = 0
        plan_start_idx = 0
        traj_lengths = [len(traj) for traj in trajectories]
        
        for i, traj_length in enumerate(traj_lengths):
            traj_end_idx = traj_start_idx + traj_length
            normed_traj = normed_all_trajectories[traj_start_idx:traj_end_idx]
            
            if self.use_plan:
                from genMoPlan.utils.plan import combine_plan_trajectory

                plan_length = plan_lengths[i]
                plan_end_idx = plan_start_idx + plan_length
                normed_plan = normed_all_plans[plan_start_idx:plan_end_idx]

                assert len(normed_plan) == len(normed_traj), "Plan and trajectory lengths do not match"
                
                normed_traj = combine_plan_trajectory(normed_plan, normed_traj)
                plan_start_idx = plan_end_idx
            
            normed_trajectories.append(torch.FloatTensor(normed_traj))
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
            return torch.zeros(0)
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

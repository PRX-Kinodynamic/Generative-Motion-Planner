from collections import namedtuple
from tqdm import tqdm

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
        stride=1,
        normalizer="LimitsNormalizer",
        normalizer_params={},
        preprocess_fns=(),
        preprocess_kwargs={},
        dataset_size=None,
        max_path_length=2000,
        use_padding=False,
        use_plans=False,
        manifold=False
    ):
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.stride = stride
        self.use_plans = use_plans
        self.manifold = manifold

        self.use_history_padding = use_padding  # check this change

        if dataset is None:
            raise ValueError("dataset not specified")
        
        if use_plans:
            from mg_diffuse.utils.plan import load_plans, combine_plans_trajectories
            data = load_plans(dataset, dataset_size)
            trajectories = data["trajectories"]
            plans = data["plans"]
            trajectories = combine_plans_trajectories(plans, trajectories)
        else:

            from mg_diffuse.utils.plan import load_plans
            data = load_plans(dataset, dataset_size)
            trajectories = data["trajectories"]
            plans = data["plans"]


            # FOR PENDULUM 
            # from mg_diffuse.utils.trajectory import load_trajectories
            # trajectories = load_trajectories(dataset, dataset_size)

        # for preprocess_fn in preprocess_fns:
        #     trajectories = preprocess_fn(trajectories, **preprocess_kwargs)

        # trajectories = np.array(trajectories, dtype=np.float32)

        path_lengths = [len(trajectory) for trajectory in trajectories]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        if manifold:

            if self.use_plans:
                # Merge trajectory and plan normalization parameters
                merged_traj_plan_norm_params = {
                    "mins": normalizer_params["trajectory"]["mins"] + normalizer_params["plan"]["mins"],
                    "maxs": normalizer_params["trajectory"]["maxs"] + normalizer_params["plan"]["maxs"],
                }
            else:
                merged_traj_plan_norm_params = normalizer_params["trajectory"]
            self.trajectory_normalizer = normalizer(manifold, trajectories, params=merged_traj_plan_norm_params)
            
        else:
            self.trajectory_normalizer = normalizer(trajectories, normalizer_params["trajectory"])
            if use_plans:
                self.plan_normalizer = normalizer(plans,  params=normalizer_params["plan"])

            

        self.indices = self.make_indices(path_lengths)

        self.observation_dim = trajectories[0].shape[-1]
        self.trajectories = trajectories
        self.n_episodes = trajectories[0].shape[0]

        if self.use_plans:
            self.plans = plans
            self.plan_dim = plans[0].shape[-1]

        # self.observation_dim = trajectories[0].shape[-1]
        # self.trajectories = trajectories
        # self.n_episodes = trajectories[0].shape[0]
        self.path_lengths = path_lengths

        self.normalize()

    # def normalize(self):
    #     """
    #     normalize fields that will be predicted by the diffusion model
    #     """
    #     array = self.trajectories.reshape(self.n_episodes * self.max_path_length, -1)
    #     normed = self.normalizer(array)
    #     self.normed_trajectories = normed.reshape(
    #         self.n_episodes, self.max_path_length, -1
    #     )

    def normalize(self):
        """
        normalize fields that will be predicted by the diffusion model
        """
        normed_trajectories = []

        plans_flag = ""
        if self.use_plans:
            normed_plans = []
            plans_flag = "with plans"

        
        # Process trajectories and plans in a single loop
        print(f"[ datasets/plan ] Normalizing trajectories {plans_flag}")
        for i in tqdm(range(len(self.trajectories))):
            # Normalize trajectory
            normed_traj = self.trajectory_normalizer(self.trajectories[i])
            normed_traj_tensor = torch.FloatTensor(normed_traj)
            normed_trajectories.append(normed_traj_tensor)
            
            if self.use_plans and not self.manifold:
                # Normalize plan
                normed_plan = self.plan_normalizer(self.plans[i])
                normed_plan_tensor = torch.FloatTensor(normed_plan)
                normed_plans.append(normed_plan_tensor)
        
        self.normed_trajectories = normed_trajectories

        if self.use_plans and not self.manifold:
            self.normed_plans = normed_plans

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

        if type(self.normed_trajectories) == list:
            trajectory = self.normed_trajectories[path_ind][start:end:self.stride]
        else:
            trajectory = self.normed_trajectories[path_ind, start:end:self.stride]

        if self.use_history_padding and len(trajectory) < self.horizon:
            padding_length = self.horizon - len(trajectory)
            padding = np.tile(trajectory[-1], (padding_length, 1))
            trajectory = np.concatenate([trajectory, padding], axis=0)

        assert len(trajectory) == self.horizon, "Trajectory length is not as expected"

        conditions = self.get_conditions(trajectory)
        batch = Batch(trajectories=trajectory, conditions=conditions)
        return batch
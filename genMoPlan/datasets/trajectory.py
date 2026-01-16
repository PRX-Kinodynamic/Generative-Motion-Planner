import torch
import numpy as np
from typing import List, Optional, Callable

from tqdm import tqdm

from genMoPlan.datasets.normalization import *
from genMoPlan.datasets.utils import EMPTY_DICT, apply_padding, make_indices, DataSample, NONE_TENSOR, FinalStateDataSample
from genMoPlan.utils.constants import (
    MASK_ON,
    MASK_OFF,
    PADDING_STRATEGY_ZEROS,
    PADDING_STRATEGY_FIRST,
    PADDING_STRATEGY_LAST,
    PADDING_STRATEGY_MIRROR,
    VALID_PADDING_STRATEGIES,
    DEFAULT_HISTORY_MASK_PADDING_VALUE,
    DEFAULT_HISTORY_PADDING_STRATEGY,
    validate_padding_strategy,
)
from genMoPlan.utils import load_trajectories, compute_actual_length, warn_stride_horizon_length
from genMoPlan.utils.arrays import batchify


class TrajectoryDataset(torch.utils.data.Dataset):
    normed_trajectories = None

    def __init__(
        self,
        dataset: str,
        read_trajectory_fn: Callable,
        horizon_length: int = 31,
        history_length: int = 1,
        stride: int = 1,
        observation_dim: int = None,
        trajectory_normalizer: str = "LimitsNormalizer",
        normalizer_params: dict = {},
        trajectory_preprocess_fns: tuple = (),
        preprocess_kwargs: dict = {},
        dataset_size: int = None,
        fnames: Optional[List[str]] = None,
        use_horizon_padding: bool = False,
        use_history_padding: bool = False,
        use_history_mask: bool = False,
        history_padding_anywhere: bool = True,
        history_padding_k = "k1",
        history_mask_padding_value: str = None,  # NEW: Padding strategy when using history masking
        history_padding_strategy: str = None,    # NEW: Padding strategy when using history padding
        is_history_conditioned: bool = True, # Otherwise it is provided as a query that is not predicted by the model
        is_validation: bool = False,
        perform_final_state_evaluation: bool = False,
        **kwargs,
    ):
        """
        TrajectoryDataset for loading and processing trajectory data.

        Args:
            dataset: Name of the dataset
            read_trajectory_fn: Function to read trajectory files
            horizon_length: Length of the horizon (future) portion
            history_length: Length of the history (past) portion
            stride: Stride for subsampling trajectories
            observation_dim: Dimension of observations
            trajectory_normalizer: Normalizer class name
            normalizer_params: Parameters for the normalizer
            trajectory_preprocess_fns: Preprocessing functions to apply
            preprocess_kwargs: Kwargs for preprocessing functions
            dataset_size: Maximum number of trajectories to load
            fnames: Specific filenames to load
            use_horizon_padding: Whether to pad horizon if too short
            use_history_padding: Whether to pad history without masking
            use_history_mask: Whether to use masking for variable-length history
            history_padding_anywhere: Allow padding at any position
            history_padding_k: Configuration for k values in padding
            history_mask_padding_value: Padding strategy when using history masking
                - "zeros" (default): Pad with zeros
                - "first": Pad with first available state
                - "last": Pad with last available state
                - "mirror": Pad with reflected sequence
            history_padding_strategy: Padding strategy when using history padding
                - "first" (default): Pad with first available state
                - "last": Pad with last available state
                - "zeros": Pad with zeros
                - "mirror": Pad with reflected sequence
            is_history_conditioned: Whether history is used for conditioning
            is_validation: Whether this is a validation dataset
            perform_final_state_evaluation: Whether to prepare final state evaluation data
        """
        self.horizon_length = horizon_length
        self.history_length = history_length
        self.stride = stride
        self.use_horizon_padding = use_horizon_padding
        self.use_history_padding = use_history_padding
        self.use_history_mask = use_history_mask
        self.history_padding_anywhere = history_padding_anywhere
        self.history_padding_k = history_padding_k
        self.is_history_conditioned = is_history_conditioned
        self.observation_dim = observation_dim

        # Warn if stride has no effect on horizon
        warn_stride_horizon_length(horizon_length, stride, context="TrajectoryDataset")

        # Set padding strategy defaults based on mode
        if history_mask_padding_value is None:
            self.history_mask_padding_value = DEFAULT_HISTORY_MASK_PADDING_VALUE
        else:
            validate_padding_strategy(history_mask_padding_value, context="history_mask_padding_value")
            self.history_mask_padding_value = history_mask_padding_value

        if history_padding_strategy is None:
            self.history_padding_strategy = DEFAULT_HISTORY_PADDING_STRATEGY
        else:
            validate_padding_strategy(history_padding_strategy, context="history_padding_strategy")
            self.history_padding_strategy = history_padding_strategy

        if perform_final_state_evaluation:
            assert is_validation, "perform_final_state_evaluation is only supported for validation dataset"

        trajectories = self._load_data(
            dataset,
            read_trajectory_fn,
            dataset_size, 
            fnames,
            trajectory_preprocess_fns, 
            preprocess_kwargs,
            is_validation,
        )

        traj_lengths = [len(trajectory) for trajectory in trajectories]
        
        self.indices = make_indices(
            traj_lengths,
            self.history_length,
            self.use_history_padding,
            self.horizon_length,
            self.use_horizon_padding,
            self.stride,
            use_history_mask=self.use_history_mask,
            history_padding_anywhere=self.history_padding_anywhere,
            history_padding_k=self.history_padding_k,
        )

        

        self.n_episodes = len(trajectories)
        self.traj_lengths = traj_lengths

        self.normed_trajectories = self._normalize(
            trajectories, 
            trajectory_normalizer, 
            normalizer_params
        )

        if perform_final_state_evaluation:
            self.eval_data = self._prepare_evaluation_data(self.normed_trajectories)


    def _load_data(
        self, 
        dataset, 
        read_trajectory_fn, 
        dataset_size, 
        fnames, 
        trajectory_preprocess_fns, 
        preprocess_kwargs, 
        is_validation,
    ):
        trajectories = load_trajectories(
            dataset, 
            read_trajectory_fn, 
            dataset_size, 
            fnames=fnames, 
            load_reverse=is_validation,
        )
        
        for trajectory_preprocess_fn in trajectory_preprocess_fns:
            trajectories = trajectory_preprocess_fn(trajectories, **preprocess_kwargs.get("trajectory", {}))
        return trajectories

    def _normalize(self, trajectories, trajectory_normalizer=None, normalizer_params=None):
        """
        normalize fields that will be predicted

        First, aggregate all trajectories into a single array
        Then, normalize the aggregated array
        Then, split the normalized array back into individual trajectories.
        """
        if trajectory_normalizer is None:
            self.normalizer = None
            return [
                torch.FloatTensor(trajectory)
                for trajectory in trajectories
            ]

        print(f"[ datasets/trajectory ] Normalizing trajectories")

        all_trajectories = np.concatenate(trajectories, axis=0)

        if type(trajectory_normalizer) == str:
            trajectory_normalizer = eval(trajectory_normalizer)

        normalizer_instance = trajectory_normalizer(X=trajectories, params=normalizer_params["trajectory"])
        normed_all_trajectories = normalizer_instance(all_trajectories)

        # Store the normalizer for later use (e.g., unnormalized loss computation)
        self.normalizer = normalizer_instance

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

    
    def _prepare_evaluation_data(self, trajectories):
        history_end = compute_actual_length(self.history_length, self.stride)

        histories = []
        final_states = []
        max_path_length = 0

        for trajectory in trajectories:
            history = trajectory[:history_end:self.stride]
            final_state = trajectory[-1]

            histories.append(history)
            final_states.append(final_state)

            # Track the maximum path length across all trajectories
            max_path_length = max(max_path_length, len(trajectory))

        histories = torch.stack(histories)
        final_states = torch.stack(final_states)

        return FinalStateDataSample(histories=histories, final_states=final_states, max_path_length=max_path_length)

    def get_conditions(self, history):
        """
        conditions on current observation for planning
        """
        if not self.is_history_conditioned:
            return EMPTY_DICT
        
        if history.ndim == 2: # history_length x observation_dim
            return dict(enumerate(history))
        else: # batch_size x history_length x observation_dim
            return dict(enumerate(history.transpose(1, 0)))

    def get_global_query(self, history):
        """
        query on current observation for planning
        """
        if self.is_history_conditioned:
            return NONE_TENSOR
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

        mask = NONE_TENSOR

        if self.use_history_padding:
            # Apply history padding using the configured strategy
            history = apply_padding(
                history,
                self.history_length,
                pad_left=True,
                strategy=self.history_padding_strategy,
            )
        elif self.use_history_mask:
            # Apply history masking: pad with configured strategy and create mask
            # Mask convention: MASK_OFF=1 (valid/present), MASK_ON=0 (masked/missing)
            provided = len(history)
            pad = self.history_length - provided
            if pad > 0:
                # Apply padding using configured strategy
                history = apply_padding(
                    history,
                    self.history_length,
                    pad_left=True,
                    strategy=self.history_mask_padding_value,
                )

            # Build mask for full prediction length (history + horizon)
            # Start with all positions valid (MASK_OFF=1)
            hist_mask = torch.full(
                (self.history_length,),
                MASK_OFF,
                dtype=torch.float32,
                device=history.device
            )
            if pad > 0:
                # Mark padded positions as masked (MASK_ON=0)
                hist_mask[:pad] = MASK_ON

            # Horizon positions are always valid (never masked)
            hor_mask = torch.full(
                (self.horizon_length,),
                MASK_OFF,
                dtype=torch.float32,
                device=history.device
            )
            mask = torch.cat([hist_mask, hor_mask], dim=0)

        if self.use_horizon_padding:
            if len(horizon) > 0:
                pad_value = horizon[-1]
            else:
                pad_value = history[-1]

            horizon = apply_padding(horizon, self.horizon_length, pad_left=False, pad_value=pad_value)

        assert len(history) == self.history_length, f"History length is {len(history)}, expected {self.history_length}"
        assert len(horizon) == self.horizon_length, f"Horizon length is {len(horizon)}, expected {self.horizon_length}"
        

        trajectory = self.get_trajectory(history, horizon)
        conditions = self.get_conditions(history)
        global_query = self.get_global_query(history)

        data_sample = DataSample(
            trajectory=trajectory,
            conditions=conditions,
            global_query=global_query,
            local_query=NONE_TENSOR,
            mask=mask,
        )

        return data_sample

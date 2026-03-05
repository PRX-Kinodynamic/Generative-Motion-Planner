import torch
import numpy as np
from typing import List, Optional, Callable

from tqdm import tqdm

from genMoPlan.datasets.normalization import *
from genMoPlan.datasets.utils import EMPTY_DICT, apply_padding, make_indices, DataSample, NONE_TENSOR, FinalStateDataSample
from genMoPlan.utils import load_trajectories, compute_actual_length
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
        is_history_conditioned: bool = True, # Otherwise it is provided as a query that is not predicted by the model
        is_validation: bool = False,
        perform_final_state_evaluation: bool = False,
        rollout_steps: int = 1,
        rollout_target_mode: str = "gt_future",
        adaptive_rollout: dict = None,
        **kwargs,
    ):
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
        self.rollout_steps = rollout_steps
        self.rollout_target_mode = rollout_target_mode
        self.adaptive_rollout = adaptive_rollout or {}

        sampled_horizon_length = self.horizon_length
        self.adaptive_base_span = int(self.adaptive_rollout.get("base_span", sampled_horizon_length))
        self.adaptive_max_span = int(self.adaptive_rollout.get("max_span", self.adaptive_base_span))
        if self.adaptive_base_span <= 0:
            raise ValueError(f"adaptive_rollout.base_span must be > 0, got {self.adaptive_base_span}")
        if self.adaptive_max_span < self.adaptive_base_span:
            raise ValueError(
                f"adaptive_rollout.max_span ({self.adaptive_max_span}) must be >= base_span ({self.adaptive_base_span})"
            )
        if self.adaptive_max_span > self.horizon_length:
            raise ValueError(
                f"adaptive_rollout.max_span ({self.adaptive_max_span}) must be <= horizon_length ({self.horizon_length})"
            )
        self.num_rollout_targets = max(0, self.rollout_steps - 1)
        self.adaptive_shared_shifts = self._build_shared_shifts(self.num_rollout_targets)

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
            rollout_steps=self.rollout_steps,
            rollout_target_mode=self.rollout_target_mode,
            adaptive_rollout=self.adaptive_rollout,
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
        
        # Debug: print first trajectory info
        print(f"[ datasets/trajectory ] normed_all_trajectories shape: {normed_all_trajectories.shape}")
        print(f"[ datasets/trajectory ] First 3 traj_lengths: {traj_lengths[:3]}")
        
        for traj_length in traj_lengths:
            traj_end_idx = traj_start_idx + traj_length
            normed_traj = normed_all_trajectories[traj_start_idx:traj_end_idx]
            
            # Use torch.from_numpy for reliable tensor creation
            normed_traj_tensor = torch.from_numpy(np.ascontiguousarray(normed_traj).astype(np.float32))
            normed_trajectories.append(normed_traj_tensor)
            traj_start_idx = traj_end_idx
        
        # Debug: verify first trajectory shape
        if normed_trajectories:
            print(f"[ datasets/trajectory ] First normed trajectory shape: {normed_trajectories[0].shape}")
        
        return normed_trajectories

    
    def _prepare_evaluation_data(self, trajectories):
        history_end = compute_actual_length(self.history_length, self.stride)

        histories = []
        final_states = []
        full_trajectories = []

        for trajectory in trajectories:
            history = trajectory[:history_end:self.stride]
            final_state = trajectory[-1]

            histories.append(history)
            final_states.append(final_state)
            full_trajectories.append(trajectory)

        histories = torch.stack(histories)
        final_states = torch.stack(final_states)
        # Store full trajectories as a list (they may have different lengths)
        # Or pad them if needed - for now keep as list
        
        return FinalStateDataSample(histories=histories, final_states=final_states, full_trajectories=full_trajectories)

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

    def _build_shared_shifts(self, num_rollout_targets):
        num_transitions = max(0, num_rollout_targets - 1)
        if num_transitions == 0:
            return []

        schedule = self.adaptive_rollout.get("shared_shift_schedule", {"type": "arithmetic", "start": 1, "delta": 0})
        if isinstance(schedule, (list, tuple)):
            shifts = [int(v) for v in schedule]
        elif isinstance(schedule, dict):
            schedule_type = schedule.get("type", "arithmetic")
            if schedule_type == "arithmetic":
                start = int(schedule.get("start", 1))
                delta = int(schedule.get("delta", 0))
                shifts = [start + delta * k for k in range(num_transitions)]
            elif schedule_type == "explicit":
                shifts = [int(v) for v in schedule.get("values", [])]
            else:
                raise ValueError(f"Unsupported shared_shift_schedule type: {schedule_type}")
        else:
            raise ValueError("shared_shift_schedule must be dict, list, or tuple")

        if len(shifts) != num_transitions:
            raise ValueError(
                f"shared_shift_schedule must have {num_transitions} values for rollout_steps={self.rollout_steps}, got {len(shifts)}"
            )
        if any(v < 0 for v in shifts):
            raise ValueError(f"shared_shift_schedule values must be non-negative, got {shifts}")
        if any(v > self.horizon_length for v in shifts):
            raise ValueError(f"shared_shift_schedule values must be <= horizon_length ({self.horizon_length}), got {shifts}")
        return shifts

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
            history = apply_padding(history, self.history_length, pad_left=True)
        elif self.use_history_mask:
            # Left-pad with zeros up to history_length and create missing-mask (1 for missing, 0 for provided)
            provided = len(history)
            pad = self.history_length - provided
            if pad > 0:
                zeros = torch.zeros((pad, history.shape[-1]), dtype=history.dtype, device=history.device)
                history = torch.cat([zeros, history], dim=0)
            # Build mask for full prediction length (history + horizon)
            hist_missing = torch.zeros((self.history_length,), dtype=torch.float32, device=history.device)
            if pad > 0:
                hist_missing[:pad] = 1.0
            hor_missing = torch.zeros((self.horizon_length,), dtype=torch.float32, device=history.device)
            mask = torch.cat([hist_missing, hor_missing], dim=0)

        if self.use_horizon_padding:
            if len(horizon) > 0:
                pad_value = horizon[-1]
            else:
                pad_value = history[-1]

            horizon = apply_padding(horizon, self.horizon_length, pad_left=False, pad_value=pad_value)

        assert len(history) == self.history_length, f"History length is {len(history)}, expected {self.history_length}"
        assert len(horizon) == self.horizon_length, f"Horizon length is {len(horizon)}, expected {self.horizon_length}"
        
        # Generate rollout_targets if rollout_steps > 1
        rollout_targets = NONE_TENSOR
        if self.rollout_steps > 1 and self.rollout_target_mode == "adaptive_stride":
            num_targets = self.num_rollout_targets
            if num_targets > 0:
                raw_target_span = compute_actual_length(self.adaptive_base_span, self.stride)

                # Offsets are in sampled-step units.
                # offset[0] = 0 (first rollout target aligns with current horizon),
                # then each subsequent target shifts by shared_shift_schedule.
                offsets = [0]
                cumulative = 0
                for shift in self.adaptive_shared_shifts:
                    cumulative += shift
                    offsets.append(cumulative)
                offsets = offsets[:num_targets]

                future_targets = []
                target_lengths = []
                valid_masks = []
                max_span = self.adaptive_max_span

                for offset in offsets:
                    future_start = horizon_start + offset * self.stride
                    future_end = future_start + raw_target_span
                    future_chunk = trajectory_data[future_start:future_end:self.stride]

                    chunk_len = min(len(future_chunk), self.adaptive_base_span)
                    target_lengths.append(chunk_len)

                    if len(future_chunk) < max_span:
                        if len(future_chunk) > 0:
                            pad_value = future_chunk[-1]
                        else:
                            pad_value = horizon[-1] if len(horizon) > 0 else history[-1]
                        future_chunk = apply_padding(future_chunk, max_span, pad_left=False, pad_value=pad_value)
                    elif len(future_chunk) > max_span:
                        future_chunk = future_chunk[:max_span]

                    future_targets.append(future_chunk)

                    valid_mask = torch.zeros(max_span, dtype=torch.bool, device=trajectory_data.device)
                    valid_len = min(chunk_len, max_span)
                    if valid_len > 0:
                        valid_mask[:valid_len] = True
                    valid_masks.append(valid_mask)

                if future_targets:
                    rollout_targets = {
                        "targets": torch.stack(future_targets, dim=0),  # [R, max_span, D]
                        "target_lengths": torch.tensor(target_lengths, dtype=torch.long, device=trajectory_data.device),  # [R]
                        "shared_shifts": torch.tensor(self.adaptive_shared_shifts, dtype=torch.long, device=trajectory_data.device),  # [R-1]
                        "valid_mask": torch.stack(valid_masks, dim=0),  # [R, max_span]
                    }
        elif self.rollout_steps > 1 and self.rollout_target_mode == "gt_future":
            # Raw span for a strided horizon chunk (in the unstrided trajectory index space).
            raw_horizon_span = compute_actual_length(self.horizon_length, self.stride)
            
            # Extract future targets for rollout steps
            future_targets = []
            for k in range(1, self.rollout_steps):
                # Each horizon chunk starts at `prev_end + (stride - 1)` (same convention as horizon_start).
                # Then it spans `raw_horizon_span` raw frames, sampled every `stride`.
                gap = self.stride - 1
                future_start = horizon_end + (k - 1) * (raw_horizon_span + gap) + gap
                future_end = future_start + raw_horizon_span
                
                # Extract future chunk
                future_chunk = trajectory_data[future_start:future_end:self.stride]
                
                # Apply padding if needed (same as horizon padding logic)
                if len(future_chunk) < self.horizon_length:
                    if len(future_chunk) > 0:
                        pad_value = future_chunk[-1]
                    else:
                        pad_value = horizon[-1] if len(horizon) > 0 else history[-1]
                    future_chunk = apply_padding(future_chunk, self.horizon_length, pad_left=False, pad_value=pad_value)
                elif len(future_chunk) > self.horizon_length:
                    future_chunk = future_chunk[:self.horizon_length]
                
                future_targets.append(future_chunk)
            
            if future_targets:
                # Stack into shape [rollout_steps-1, horizon_length, D]
                rollout_targets = torch.stack(future_targets, dim=0)

        trajectory = self.get_trajectory(history, horizon)
        conditions = self.get_conditions(history)
        global_query = self.get_global_query(history)

        data_sample = DataSample(
            trajectory=trajectory,
            conditions=conditions,
            global_query=global_query,
            local_query=NONE_TENSOR,
            mask=mask,
            rollout_targets=rollout_targets,
        )

        return data_sample

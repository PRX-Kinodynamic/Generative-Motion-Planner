import warnings
import numpy as np
from typing import Callable

import torch

from genMoPlan.utils.data_processing import compute_actual_length

try:
    from flow_matching.path.scheduler import *
    from flow_matching.path import *
    from flow_matching.solver import *

    FLOW_MATCHING_AVAILABLE = True
except ImportError:
    FLOW_MATCHING_AVAILABLE = False


from genMoPlan.models.helpers import apply_conditioning
from genMoPlan.utils.arrays import torch_randn_like

from .base import GenerativeModel, Sample


def _to_float_or_none(value):
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value, precision=4, fallback="N/A"):
    numeric = _to_float_or_none(value)
    if numeric is None:
        return fallback
    return f"{numeric:.{precision}f}"


class FlowMatching(GenerativeModel):
    def __init__(
        self,
        model,
        input_dim,
        output_dim,
        prediction_length,
        history_length,
        clip_denoised=False,
        loss_type="l2",
        action_indices=None,
        has_local_query=False,
        has_global_query=False,
        stride: int = 1,
        # Flow matching specific parameters
        scheduler="CondOTScheduler",
        path="AffineProbPath",
        solver="ODESolver",
        n_fourier_features=1,
        # Rollout loss parameters
        use_rollout_loss: bool = False,
        rollout_steps: int = 1,
        rollout_weighting=None,  # Can be list of weights or dict with schedule
        rollout_loss_type: str = "l2",
        rollout_sample_kwargs: dict = None,
        rollout_operator: str = "fixed",
        adaptive_rollout: dict = None,
        **kwargs,
    ):
        if not FLOW_MATCHING_AVAILABLE:
            raise ImportError("Flow matching is not available. Please install flow_matching to use the flow matching method.")

        super().__init__(
            model=model,
            input_dim=input_dim,
            output_dim=output_dim,
            prediction_length=prediction_length,
            history_length=history_length,
            clip_denoised=clip_denoised,
            loss_type=loss_type,
            action_indices=action_indices,
            has_local_query=has_local_query,
            has_global_query=has_global_query,
            stride=stride,
            **kwargs
        )

        self.n_timesteps = 1
        self.history_length = history_length
        
        # Rollout loss configuration
        self.use_rollout_loss = use_rollout_loss
        self.rollout_steps = rollout_steps
        self.rollout_loss_type = rollout_loss_type
        self.rollout_sample_kwargs = rollout_sample_kwargs or {}
        self.rollout_operator = rollout_operator
        self.adaptive_rollout = adaptive_rollout or {}
        if self.adaptive_rollout.get("enabled", False):
            self.rollout_operator = self.adaptive_rollout.get("mode", self.rollout_operator)
        if self.rollout_operator not in ("fixed", "adaptive_stride"):
            raise ValueError(f"Unsupported rollout_operator: {self.rollout_operator}")
        
        # Process rollout weighting
        if rollout_weighting is None:
            # Default: equal weights
            self.rollout_weights = [1.0] * (rollout_steps - 1) if rollout_steps > 1 else []
        elif isinstance(rollout_weighting, (list, tuple)):
            self.rollout_weights = list(rollout_weighting)
        elif isinstance(rollout_weighting, dict):
            # Schedule-based weighting (e.g., exponential decay)
            schedule_type = rollout_weighting.get("type", "exp")
            if schedule_type == "exp":
                decay = rollout_weighting.get("decay", 0.5)
                self.rollout_weights = [decay ** k for k in range(1, rollout_steps)]
            elif schedule_type == "linear":
                start = rollout_weighting.get("start", 1.0)
                end = rollout_weighting.get("end", 0.1)
                self.rollout_weights = [start - (start - end) * k / (rollout_steps - 1) for k in range(1, rollout_steps)]
            else:
                self.rollout_weights = [1.0] * (rollout_steps - 1)
        else:
            self.rollout_weights = [1.0] * (rollout_steps - 1)

        if self.manifold is not None:
            if path != "GeodesicProbPath":
                raise ValueError("Manifold is not supported for non-geodesic paths")
            
            if solver != "RiemannianODESolver":
                raise ValueError("Riemannian solver is required for geodesic paths")
        
        # ------------------------------ Setup flow matching components ------------------------------#

        if type(scheduler) == str:
            scheduler = eval(scheduler)

        scheduler = scheduler()

        path_args = [scheduler]

        if self.manifold is not None:
            from genMoPlan.models.manifold import ProjectToTangent

            path_args.append(self.manifold)
            solver_args = [self.manifold]
            self.model = ProjectToTangent(
                model,
                manifold=self.manifold,
                input_dim=self.input_dim,
                n_fourier_features=n_fourier_features,
                use_history_mask=self.use_mask,
            )
        else:
            solver_args = []

        if type(path) == str:
            path = eval(path)
        self.path = path(*path_args)

        if type(solver) == str:
            solver = eval(solver)

        self.transformed_model = ScheduleTransformedModel(
            velocity_model=self.vector_field,
            original_scheduler = scheduler,
            new_scheduler = CondOTScheduler(),
        )

        solver_args.append(self.transformed_model)

        self.solver: Solver = solver(*solver_args)

    # --------------------------------------- vector field ----------------------------------------#

    def vector_field(self, x=None, t=None, global_query=None, local_query=None, mask=None):
        if x is None or t is None:
            raise ValueError("x and t must be provided")
        if getattr(self, "use_mask", False) and mask is None:
            raise ValueError("Mask expected but not provided")
        
        if t.ndim == 0:
            batch_size = x.shape[0]
            t = t.unsqueeze(0).repeat(batch_size)

        # The model expects parameters in the order: (x, global_query, local_query, t, mask)
        # Require mask if experiment expects it

        vector_field = self.model(x, global_query, local_query, t, mask=mask) if mask is not None else self.model(x, global_query, local_query, t)

        # Zero out the vector field for the history portion
        if self.history_length > 0:
            vector_field[:, :self.history_length, :] = 0.0
            
        return vector_field

    # ------------------------------------------ training ------------------------------------------#

    def compute_loss(self, x_target, cond, global_query=None, local_query=None, seed=None, mask=None, rollout_targets=None):
        """
        Choose a random timestep t and calculate the loss for the model.
        Optionally computes rollout loss if use_rollout_loss is enabled.
        """
        if getattr(self, "use_mask", False) and mask is None:
            raise ValueError("Mask expected by FlowMatching but not provided")
        if not getattr(self, "use_mask", False):
            mask = None

        if seed is not None:
            generator = torch.Generator(device=x_target.device).manual_seed(seed)
        else:
            generator = None

        batch_size = len(x_target)
        t = torch.rand(batch_size, device=x_target.device, generator=generator)

        x_noisy = torch_randn_like(x_target, generator=generator)
        
        apply_conditioning(x_noisy, cond)

        if self.manifold is not None:
            x_target = self.manifold.wrap(x_target)
            x_noisy = self.manifold.wrap(x_noisy)

        path_sample = self.path.sample(t=t, x_0=x_noisy, x_1=x_target)

        # Base flow matching loss
        loss, info = self.loss_fn(
            self.vector_field(x=path_sample.x_t, t=path_sample.t, global_query=global_query, local_query=local_query, mask=mask),
            path_sample.dx_t,
            ignore_manifold=True,
            loss_weights=self.loss_weights,
        )

        # Rollout loss computation
        if self.use_rollout_loss:
            if rollout_targets is None:
                if not hasattr(self, "_rollout_warning_printed"):
                    print(f"[WARNING] Rollout loss enabled but rollout_targets is None. Check dataset rollout_steps config.")
                    self._rollout_warning_printed = True
            elif torch.is_tensor(rollout_targets) and rollout_targets.numel() == 0:
                if not hasattr(self, "_rollout_warning_printed"):
                    print(f"[WARNING] Rollout loss enabled but rollout_targets is empty. Trajectories may be too short for rollout_steps={self.rollout_steps}.")
                    self._rollout_warning_printed = True
            elif isinstance(rollout_targets, dict) and (
                "targets" not in rollout_targets
                or rollout_targets["targets"] is None
                or (torch.is_tensor(rollout_targets["targets"]) and rollout_targets["targets"].numel() == 0)
            ):
                if not hasattr(self, "_rollout_warning_printed"):
                    print(f"[WARNING] Adaptive rollout enabled but rollout_targets['targets'] is empty.")
                    self._rollout_warning_printed = True
            else:
                if self.rollout_operator == "adaptive_stride":
                    if not isinstance(rollout_targets, dict):
                        raise ValueError("Adaptive rollout expects rollout_targets as a dict payload from dataset.")
                    rollout_loss, rollout_info = self._compute_adaptive_rollout_loss(
                        x_target, cond, global_query, local_query, mask, rollout_targets, generator
                    )
                else:
                    if not torch.is_tensor(rollout_targets):
                        raise ValueError("Fixed rollout expects rollout_targets as a tensor.")
                    rollout_loss, rollout_info = self._compute_rollout_loss(
                        x_target, cond, global_query, local_query, mask, rollout_targets, generator
                    )
                # Add rollout losses to info
                info.update(rollout_info)
                # Weighted sum: L_total = L_fm + Σ_k w_k * L_rollout_k
                # rollout_loss is a tensor, so it will be included in backpropagation
                loss = loss + rollout_loss

        return loss, info
    
    def _compute_rollout_loss(self, x_target, cond, global_query, local_query, mask, rollout_targets, generator=None):
        """
        Compute rollout loss for k=1..K rollout steps.
        
        Args:
            x_target: Ground truth trajectory [batch, prediction_length, dim]
            cond: Conditioning dictionary
            global_query: Global query tensor
            local_query: Local query tensor
            mask: Mask tensor
            rollout_targets: Ground truth targets for rollout steps [rollout_steps-1, horizon_length, dim] or [batch, rollout_steps-1, horizon_length, dim]
            generator: Random generator for reproducibility
            
        Returns:
            rollout_loss: Total weighted rollout loss
            info: Dictionary with per-step losses and other metrics
        """
        from genMoPlan.utils.data_processing import compute_actual_length
        
        device = x_target.device
        batch_size = x_target.shape[0]
        horizon_length = self.prediction_length - self.history_length
        actual_horizon_length = compute_actual_length(horizon_length, self.stride)
        
        # Handle rollout_targets shape: could be [rollout_steps-1, horizon_length, dim] or [batch, rollout_steps-1, horizon_length, dim]
        if rollout_targets.ndim == 3:
            # [rollout_steps-1, horizon_length, dim] -> expand to batch
            rollout_targets = rollout_targets.unsqueeze(0).expand(batch_size, -1, -1, -1)
        elif rollout_targets.ndim == 4:
            # [batch, rollout_steps-1, horizon_length, dim] - already batched
            pass
        else:
            raise ValueError(f"rollout_targets must have 3 or 4 dimensions, got {rollout_targets.ndim}")
        
        num_rollout_steps = rollout_targets.shape[1]  # rollout_steps - 1
        
        # Extract initial history from the end of the current target chunk so rollout predicts future chunks.
        current_history = x_target[:, -self.history_length:, :]  # [batch, history_length, dim]
        
        rollout_losses = []
        rollout_info = {}
        
        # Get default sample kwargs
        sample_kwargs = {
            'n_timesteps': self.rollout_sample_kwargs.get('n_timesteps', 5),
            'integration_method': self.rollout_sample_kwargs.get('integration_method', 'euler'),
        }
        
        # Subsample batch if specified (to control compute)
        batch_frac = self.rollout_sample_kwargs.get('batch_frac', 1.0)
        if batch_frac < 1.0:
            n_samples = max(1, int(batch_size * batch_frac))
            sample_indices = torch.randperm(batch_size, device=device, generator=generator)[:n_samples]
        else:
            sample_indices = torch.arange(batch_size, device=device)
        
        # Iterate over rollout steps
        for k in range(num_rollout_steps):
            # Get conditions from current history (only for sampled indices)
            if len(sample_indices) < batch_size:
                # Use only sampled indices for generation
                sampled_history = current_history[sample_indices]
                current_cond = self._get_conditions_from_history(sampled_history)
            else:
                current_cond = self._get_conditions_from_history(current_history)
            
            # Generate next horizon chunk (differentiable) - only for sampled indices
            shape = (len(sample_indices), self.prediction_length, self.input_dim)
            sol = self.conditional_sample_grad(
                current_cond,
                shape,
                global_query=global_query[sample_indices] if global_query is not None else None,
                local_query=local_query[sample_indices] if local_query is not None else None,
                mask=mask[sample_indices] if mask is not None else None,
                seed=None,  # Use different seed per step for diversity
                **sample_kwargs
            )
            
            # Extract predicted horizon (skip history)
            pred_horizon = sol.trajectories[:, self.history_length:, :]  # [batch_subset, horizon_length, dim]
            
            # Get ground truth horizon for this rollout step
            gt_horizon = rollout_targets[sample_indices, k, :, :]  # [batch_subset, horizon_length, dim]
            
            # Compute loss between predicted and GT horizon
            if self.rollout_loss_type == "l2":
                step_loss, step_info = self.loss_fn(pred_horizon, gt_horizon)
            elif self.rollout_loss_type == "manifold" and self.manifold is not None:
                # Use manifold distance
                step_loss = self.manifold.dist(pred_horizon, gt_horizon).mean()
                step_info = {'rollout_manifold_loss': step_loss.item()}
            else:
                step_loss, step_info = self.loss_fn(pred_horizon, gt_horizon)
            
            # Apply weight for this step
            weight = self.rollout_weights[k] if k < len(self.rollout_weights) else 1.0
            weighted_step_loss = weight * step_loss
            
            rollout_losses.append(weighted_step_loss)
            
            # Store per-step info
            rollout_info[f'rollout_{k+1}_loss'] = step_loss.item()
            rollout_info[f'rollout_{k+1}_weighted_loss'] = weighted_step_loss.item()
            if step_info:
                for key, val in step_info.items():
                    if isinstance(val, (int, float)):
                        rollout_info[f'rollout_{k+1}_{key}'] = val
                    elif hasattr(val, 'item'):
                        rollout_info[f'rollout_{k+1}_{key}'] = val.item()
            
            # Update history for next step: use last history_length states from prediction
            # This is where error accumulation happens
            if len(sample_indices) < batch_size:
                # Expand history to full batch: use predicted history for sampled, GT for others
                new_history = current_history.clone()  # Start with current history
                new_history[sample_indices] = sol.trajectories[:, -self.history_length:, :]
                # For non-sampled indices, advance using GT for the current step.
                non_sampled_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                non_sampled_mask[sample_indices] = False
                if non_sampled_mask.any():
                    gt_step = rollout_targets[non_sampled_mask, k, :, :]  # [n, horizon_length, dim]
                    cat = torch.cat([current_history[non_sampled_mask], gt_step], dim=1)
                    new_history[non_sampled_mask] = cat[:, -self.history_length:, :]
                current_history = new_history
            else:
                current_history = sol.trajectories[:, -self.history_length:, :]  # [batch, history_length, dim]
        
        # Sum all rollout losses
        total_rollout_loss = sum(rollout_losses)
        rollout_info['rollout_total_unweighted'] = sum([rollout_info[f'rollout_{k+1}_loss'] for k in range(num_rollout_steps)])
        rollout_info['rollout_weighted_loss'] = total_rollout_loss.item()
        
        return total_rollout_loss, rollout_info

    def _compute_masked_step_loss(self, pred, target, valid_mask):
        """
        Compute loss on valid timesteps only.

        Args:
            pred: [batch, horizon, dim]
            target: [batch, horizon, dim]
            valid_mask: [batch, horizon] boolean mask
        """
        if valid_mask is None:
            return self.loss_fn(pred, target)

        valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(pred)
        if not valid_mask_expanded.any():
            zero_loss = pred.sum() * 0.0
            return zero_loss, {"valid_points": 0, "valid_fraction": 0.0}

        pred_valid = pred[valid_mask_expanded].view(-1, pred.shape[-1])
        target_valid = target[valid_mask_expanded].view(-1, target.shape[-1])
        step_loss, step_info = self.loss_fn(pred_valid, target_valid)

        valid_points = int(valid_mask.sum().item())
        total_points = int(valid_mask.numel())
        step_info["valid_points"] = valid_points
        step_info["valid_fraction"] = valid_points / total_points if total_points > 0 else 0.0
        return step_loss, step_info

    def _compute_adaptive_rollout_loss(self, x_target, cond, global_query, local_query, mask, rollout_targets, generator=None):
        """
        Compute adaptive-stride rollout loss with shared anchor/prediction shifts.
        """
        device = x_target.device
        batch_size = x_target.shape[0]
        horizon_length = self.prediction_length - self.history_length

        targets = rollout_targets.get("targets", None)
        target_lengths = rollout_targets.get("target_lengths", None)
        shared_shifts = rollout_targets.get("shared_shifts", None)
        valid_mask = rollout_targets.get("valid_mask", None)

        if targets is None or target_lengths is None or valid_mask is None:
            raise ValueError("Adaptive rollout expects rollout_targets with keys: targets, target_lengths, valid_mask.")

        if targets.ndim == 3:
            targets = targets.unsqueeze(0).expand(batch_size, -1, -1, -1)
        elif targets.ndim != 4:
            raise ValueError(f"Adaptive rollout targets must have 3 or 4 dims, got {targets.ndim}.")

        if target_lengths.ndim == 1:
            target_lengths = target_lengths.unsqueeze(0).expand(batch_size, -1)
        elif target_lengths.ndim != 2:
            raise ValueError(f"Adaptive target_lengths must have 1 or 2 dims, got {target_lengths.ndim}.")

        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif valid_mask.ndim != 3:
            raise ValueError(f"Adaptive valid_mask must have 2 or 3 dims, got {valid_mask.ndim}.")

        if shared_shifts is not None:
            if shared_shifts.ndim == 1:
                shared_shifts = shared_shifts.unsqueeze(0).expand(batch_size, -1)
            elif shared_shifts.ndim != 2:
                raise ValueError(f"Adaptive shared_shifts must have 1 or 2 dims, got {shared_shifts.ndim}.")

        num_rollout_steps = targets.shape[1]
        max_span = targets.shape[2]
        if max_span > horizon_length:
            raise ValueError(
                f"Adaptive max span ({max_span}) exceeds model horizon length ({horizon_length})."
            )

        sample_kwargs = {
            "n_timesteps": self.rollout_sample_kwargs.get("n_timesteps", 5),
            "integration_method": self.rollout_sample_kwargs.get("integration_method", "euler"),
        }

        batch_frac = self.rollout_sample_kwargs.get("batch_frac", 1.0)
        if batch_frac < 1.0:
            n_samples = max(1, int(batch_size * batch_frac))
            sample_indices = torch.randperm(batch_size, device=device, generator=generator)[:n_samples]
        else:
            sample_indices = torch.arange(batch_size, device=device)

        targets = targets[sample_indices]
        target_lengths = target_lengths[sample_indices]
        valid_mask = valid_mask[sample_indices]
        if shared_shifts is not None:
            shared_shifts = shared_shifts[sample_indices]

        current_history = x_target[sample_indices, :self.history_length, :]
        local_query_subset = local_query[sample_indices] if local_query is not None else None
        global_query_subset = global_query[sample_indices] if global_query is not None else None
        mask_subset = mask[sample_indices] if mask is not None else None

        rollout_losses = []
        rollout_info = {}
        unweighted_sum = 0.0

        for k in range(num_rollout_steps):
            current_cond = self._get_conditions_from_history(current_history)
            shape = (len(sample_indices), self.prediction_length, self.input_dim)
            sol = self.conditional_sample_grad(
                current_cond,
                shape,
                global_query=global_query_subset,
                local_query=local_query_subset,
                mask=mask_subset,
                seed=None,
                **sample_kwargs,
            )

            pred_horizon = sol.trajectories[:, self.history_length:, :]
            pred_step = pred_horizon[:, :max_span, :]
            gt_step = targets[:, k, :, :]
            step_valid_mask = valid_mask[:, k, :].bool()

            step_loss, step_info = self._compute_masked_step_loss(pred_step, gt_step, step_valid_mask)

            weight = self.rollout_weights[k] if k < len(self.rollout_weights) else 1.0
            weighted_step_loss = weight * step_loss
            rollout_losses.append(weighted_step_loss)

            step_loss_item = step_loss.item()
            unweighted_sum += step_loss_item
            rollout_info[f"adaptive_rollout_step_{k+1}_loss"] = step_loss_item
            rollout_info[f"adaptive_rollout_step_{k+1}_weighted_loss"] = weighted_step_loss.item()
            rollout_info[f"adaptive_rollout_span_{k+1}"] = target_lengths[:, k].float().mean().item()

            if shared_shifts is not None and k < shared_shifts.shape[1]:
                step_shift = shared_shifts[:, k].float().mean().item()
            else:
                step_shift = 0.0
            rollout_info[f"adaptive_rollout_shared_shift_{k+1}"] = step_shift

            for key, val in step_info.items():
                if isinstance(val, (int, float)):
                    rollout_info[f"adaptive_rollout_step_{k+1}_{key}"] = float(val)
                elif hasattr(val, "item"):
                    rollout_info[f"adaptive_rollout_step_{k+1}_{key}"] = val.item()

            if k >= num_rollout_steps - 1:
                continue

            if shared_shifts is None:
                shift_values = torch.ones(len(sample_indices), dtype=torch.long, device=device)
            elif k < shared_shifts.shape[1]:
                shift_values = shared_shifts[:, k].to(dtype=torch.long, device=device)
            else:
                shift_values = torch.ones(len(sample_indices), dtype=torch.long, device=device)

            max_start = sol.trajectories.shape[1] - self.history_length
            shift_values = torch.clamp(shift_values, min=0, max=max_start)

            if torch.all(shift_values == shift_values[0]):
                start = int(shift_values[0].item())
                current_history = sol.trajectories[:, start:start + self.history_length, :]
            else:
                history_idx = torch.arange(self.history_length, device=device).view(1, -1)
                gather_idx = shift_values.view(-1, 1) + history_idx
                gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, self.input_dim)
                current_history = torch.gather(sol.trajectories, dim=1, index=gather_idx)

        if rollout_losses:
            total_rollout_loss = sum(rollout_losses)
        else:
            total_rollout_loss = x_target.sum() * 0.0

        rollout_info["adaptive_rollout_total_unweighted"] = unweighted_sum
        rollout_info["adaptive_rollout_total_weighted"] = total_rollout_loss.item()
        return total_rollout_loss, rollout_info
    
    def _get_conditions_from_history(self, history):
        """
        Convert history tensor to conditions dictionary format.
        history: [batch, history_length, dim]
        """
        if history.ndim == 2:
            # Single sample: [history_length, dim]
            return dict(enumerate(history))
        else:
            # Batch: [batch, history_length, dim]
            return dict(enumerate(history.transpose(1, 0)))

    # ------------------------------------------ rollout loss ------------------------------------------#
    
    def conditional_sample_grad(self, cond, shape, global_query=None, local_query=None, n_timesteps=5, integration_method="euler", return_chain=False, n_intermediate_steps=0, seed=None, mask=None, **kwargs) -> Sample:
        """
        Generate samples with gradients enabled (for training rollout loss).
        Same as conditional_sample but without @torch.no_grad().
        
        Args:
            cond: Conditioning information that will be applied to the samples
            shape: Shape of the output samples
            global_query: Optional global query tensor for conditional generation
            local_query: Optional local query tensor for conditional generation
            n_timesteps: Number of timesteps to use for the ODE solver
            integration_method: Integration method to use ("midpoint", "euler", etc.)
            return_chain: Whether to return intermediate states in the sampling chain
            n_intermediate_steps: Number of intermediate steps to save (if return_chain=True)
            mask: Optional mask to apply to the samples
            **kwargs: Additional arguments passed to the solver
            
        Returns:
            Sample: An object containing the generated trajectories and optional intermediate chains
        """
        device = next(self.parameters()).device

        assert n_intermediate_steps >= 0
        if getattr(self, "use_mask", False) and mask is None:
            raise ValueError("Mask expected but not provided")
        if not getattr(self, "use_mask", False):
            mask = None

        if n_intermediate_steps == 0 and return_chain:
            warnings.warn("n_intermediate_steps is 0 and return_chain is True, this will return only the noisy and final samples")

        T = torch.linspace(0, 1, n_intermediate_steps + 2)

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        x_noisy = torch.randn(shape, device=device, generator=generator)
        
        apply_conditioning(x_noisy, cond)

        if self.manifold is not None:
            x_noisy = self.manifold.wrap(x_noisy)

        model_extras = {}

        if global_query is not None:
            model_extras['global_query'] = global_query
        if local_query is not None:
            model_extras['local_query'] = local_query
        if mask is not None:
            model_extras['mask'] = mask

        # Enable gradients for rollout loss
        sol = self.solver.sample(
            x_init=x_noisy, 
            step_size=1.0/n_timesteps, 
            time_grid=T, 
            method=integration_method, 
            return_intermediates=return_chain,
            enable_grad=True,  # Enable gradients for training
            **model_extras
        )

        if return_chain:
            chains = sol
            sol = sol[-1]
        else:
            chains = None

        if self.clip_denoised:
            sol = torch.clamp(sol, -1.0, 1.0)

        if self.manifold is not None:
            sol = self.manifold.wrap(sol) # Wrap the samples to the manifold for safety
            sol = self.manifold.unwrap(sol) # Unwrap the samples to the original space

        return Sample(trajectories=sol, values=None, chains=chains)

    
    # ------------------------------------------ inference ------------------------------------------#

    @torch.no_grad()
    def conditional_sample(self, cond, shape, global_query=None, local_query=None, n_timesteps=5, integration_method="euler", return_chain=False, n_intermediate_steps=0, seed=None, mask=None, **kwargs) -> Sample:
        """
        Generate samples by running the flow matching ODE solver from noise to target.
        
        Args:
            cond: Conditioning information that will be applied to the samples
            shape: Shape of the output samples
            global_query: Optional global query tensor for conditional generation
            local_query: Optional local query tensor for conditional generation
            n_timesteps: Number of timesteps to use for the ODE solver
            integration_method: Integration method to use ("midpoint", "euler", etc.)
            return_chain: Whether to return intermediate states in the sampling chain
            n_intermediate_steps: Number of intermediate steps to save (if return_chain=True)
            mask: Optional mask to apply to the samples
            **kwargs: Additional arguments passed to the solver
            
        Returns:
            Sample: An object containing the generated trajectories and optional intermediate chains
        """

        device = next(self.parameters()).device

        assert n_intermediate_steps >= 0
        if getattr(self, "use_mask", False) and mask is None:
            raise ValueError("Mask expected but not provided")
        if not getattr(self, "use_mask", False):
            mask = None

        if n_intermediate_steps == 0 and return_chain:
            warnings.warn("n_intermediate_steps is 0 and return_chain is True, this will return only the noisy and final samples")

        T = torch.linspace(0, 1, n_intermediate_steps + 2)

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        x_noisy = torch.randn(shape, device=device, generator=generator)
        
        apply_conditioning(x_noisy, cond)

        if self.manifold is not None:
            x_noisy = self.manifold.wrap(x_noisy)

        model_extras = {}

        if global_query is not None:
            model_extras['global_query'] = global_query
        if local_query is not None:
            model_extras['local_query'] = local_query
        if mask is not None:
            model_extras['mask'] = mask

        sol = self.solver.sample(
            x_init=x_noisy, 
            step_size=1.0/n_timesteps, 
            time_grid=T, 
            method=integration_method, 
            return_intermediates=return_chain,
            **model_extras
        )

        if return_chain:
            chains = sol
            sol = sol[-1]
        else:
            chains = None

        if self.clip_denoised:
            sol = torch.clamp(sol, -1.0, 1.0)

        if self.manifold is not None:
            sol = self.manifold.wrap(sol) # Wrap the samples to the manifold for safety
            sol = self.manifold.unwrap(sol) # Unwrap the samples to the original space

        return Sample(trajectories=sol, values=None, chains=chains)
    
    # ------------------------------------------ validation ------------------------------------------#

    def _compute_real_mask(self, x, history_length=1):
        """
        Compute a mask indicating which timesteps contain real data vs padding.
        
        Padding is detected by finding consecutive identical values at the end of the horizon.
        The history portion is always considered real.
        
        Args:
            x: Trajectory tensor (batch, seq_len, dim)
            history_length: Number of history timesteps at the start (always real)
            
        Returns:
            real_mask: Boolean tensor (batch, seq_len) - True for real data, False for padding
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        
        # Initialize mask - assume all real
        real_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # History is always real (first history_length timesteps)
        # Check horizon portion for padding
        horizon_start = history_length
        
        if horizon_start >= seq_len:
            return real_mask
        
        # For each sample, find where padding starts
        # Padding = consecutive timesteps that are identical to the last timestep
        for b in range(batch_size):
            horizon = x[b, horizon_start:]  # (horizon_len, dim)
            
            if len(horizon) <= 1:
                continue
            
            # Find the first real (non-padding) timestep from the end
            # Padding means the value equals the value at the end
            last_value = horizon[-1]  # (dim,)
            
            # Check backwards from end to find where real data ends
            padding_start = len(horizon)  # Assume no padding initially
            for t in range(len(horizon) - 2, -1, -1):  # Go backwards
                # Check if this timestep equals the last value (padding)
                if torch.allclose(horizon[t], last_value, atol=1e-6):
                    padding_start = t
                else:
                    break  # Found real data, stop
            
            # Mark padding timesteps as False
            if padding_start < len(horizon):
                real_mask[b, horizon_start + padding_start:] = False
        
        return real_mask

    def validation_loss(self, x, cond, global_query=None, local_query=None, mask=None, rollout_targets=None, verbose=False, **sample_kwargs):
        if not self.has_local_query:
            local_query = None
        if not self.has_global_query:
            global_query = None

        sol = self.conditional_sample(
            cond,
            x.shape,
            global_query=global_query,
            local_query=local_query,
            verbose=False,
            return_chain=False,
            seed=self.val_seed,
            mask=mask,
            **sample_kwargs,
        )

        pred_final_state = sol.trajectories[..., -1, :]
        target_final_state = x[..., -1, :]

        final_state_loss, final_state_info = self.loss_fn(pred_final_state, target_final_state)

        # Standard loss (includes padding)
        loss, info = self.loss_fn(sol.trajectories, x, loss_weights=self.loss_weights)

        info["final_state_loss"] = final_state_loss
        
        # Add detailed per-component final state losses
        for key, val in final_state_info.items():
            info[f"final_{key}"] = val
        
        # ---- Real-data-only loss (Option 3) ----
        # Compute mask for real vs padded data
        history_length = len(cond) if isinstance(cond, dict) else 1
        real_mask = self._compute_real_mask(x, history_length=history_length)
        
        # Calculate real data statistics
        total_timesteps = real_mask.numel()
        real_timesteps = real_mask.sum().item()
        real_fraction = real_timesteps / total_timesteps if total_timesteps > 0 else 0
        
        info["real_fraction"] = real_fraction
        info["real_timesteps"] = real_timesteps
        info["total_timesteps"] = total_timesteps
        
        # Compute loss only on real data if there's any
        if real_timesteps > 0:
            # Expand mask to match dimensions (batch, seq_len, dim)
            real_mask_expanded = real_mask.unsqueeze(-1).expand_as(x)
            
            # Extract real data points
            pred_real = sol.trajectories[real_mask_expanded].view(-1, x.shape[-1])
            target_real = x[real_mask_expanded].view(-1, x.shape[-1])
            
            if len(pred_real) > 0:
                # Compute loss on real data only
                # Use manifold distance when available (e.g. torus/angular for theta), else raw L2
                if self.manifold is not None:
                    dist_real = self.manifold.dist(pred_real, target_real)  # (N, dim)
                    real_data_loss = dist_real.mean()
                    info["real_data_loss"] = real_data_loss.item()
                    if x.shape[-1] >= 4:
                        info["real_x_loss"] = dist_real[:, 0].mean().item()
                        info["real_theta_loss"] = dist_real[:, 1].mean().item()
                        info["real_x_dot_loss"] = dist_real[:, 2].mean().item()
                        info["real_theta_dot_loss"] = dist_real[:, 3].mean().item()
                else:
                    real_data_loss = torch.sqrt(((pred_real - target_real) ** 2).mean())
                    info["real_data_loss"] = real_data_loss.item()
                    if x.shape[-1] >= 4:
                        info["real_x_loss"] = torch.sqrt(((pred_real[:, 0] - target_real[:, 0]) ** 2).mean()).item()
                        info["real_theta_loss"] = torch.sqrt(((pred_real[:, 1] - target_real[:, 1]) ** 2).mean()).item()
                        info["real_x_dot_loss"] = torch.sqrt(((pred_real[:, 2] - target_real[:, 2]) ** 2).mean()).item()
                        info["real_theta_dot_loss"] = torch.sqrt(((pred_real[:, 3] - target_real[:, 3]) ** 2).mean()).item()
            else:
                info["real_data_loss"] = loss.item()  # Fallback
        else:
            info["real_data_loss"] = loss.item()  # Fallback if no real data
        
        if verbose:
            batch_size = x.shape[0]
            real_pct = real_fraction * 100
            print(f"    [Validation Batch] samples={batch_size} | "
                  f"traj_loss={_fmt(loss)} | final_state_loss={_fmt(final_state_loss)} | "
                  f"real_data_loss={_fmt(info.get('real_data_loss', 0))} ({real_pct:.1f}% real)")
            print(f"      Per-component (trajectory - ALL data): "
                  f"x={_fmt(info.get('x_loss', 0))} | "
                  f"θ={_fmt(info.get('theta_loss', 0))} | "
                  f"ẋ={_fmt(info.get('x_dot_loss', 0))} | "
                  f"θ̇={_fmt(info.get('theta_dot_loss', 0))}")
            print(f"      Per-component (trajectory - REAL data only): "
                  f"x={_fmt(info.get('real_x_loss', 0))} | "
                  f"θ={_fmt(info.get('real_theta_loss', 0))} | "
                  f"ẋ={_fmt(info.get('real_x_dot_loss', 0))} | "
                  f"θ̇={_fmt(info.get('real_theta_dot_loss', 0))}")
            print(f"      Per-component (final state): "
                  f"x={_fmt(final_state_info.get('x_loss', 0))} | "
                  f"θ={_fmt(final_state_info.get('theta_loss', 0))} | "
                  f"ẋ={_fmt(final_state_info.get('x_dot_loss', 0))} | "
                  f"θ̇={_fmt(final_state_info.get('theta_dot_loss', 0))}")

        return loss, info

    def evaluate_final_states(self, start_states: torch.Tensor, target_final_states: torch.Tensor, global_query: torch.Tensor =None, local_query: torch.Tensor =None, max_path_length: int =None, get_conditions: Callable =None, full_trajectories: list =None, **sample_kwargs):
        """
        Evaluate the final states of the model given the start states and the target final states.

        Args:
            start_states: The start states to evaluate the final states from. (batch_size, history_length, dim)
            target_final_states: The target final states to evaluate the final states from. (batch_size, dim)
            global_query: The global query to use for the model.
            local_query: The local query to use for the model.
            max_path_length: The maximum path length to use for the model.
            get_conditions: The function to get the conditions for the model.
            full_trajectories: List of full ground truth trajectories for intermediate comparison.
            **sample_kwargs: Additional arguments passed to the conditional sample method.

        Returns:
            final_state_loss: The loss for the final states.
            final_state_info: The information for the final states.
        """

        assert max_path_length is not None, "max_path_length must be provided for final state evaluation"
        assert get_conditions is not None, "get_conditions must be provided for final state evaluation"

        batch_size = target_final_states.shape[0]

        # Compute actual lengths accounting for stride
        actual_history = compute_actual_length(self.history_length, self.stride)
        horizon_length = self.prediction_length - self.history_length
        actual_horizon = compute_actual_length(horizon_length, self.stride)
        
        num_inference_steps = np.ceil((max_path_length - actual_history) / actual_horizon).astype(int)
        
        # DEBUG: Print stride and num_inference_steps to verify fix is working
        print(f"[DEBUG evaluate_final_states] stride={self.stride}, history_length={self.history_length}, "
              f"prediction_length={self.prediction_length}, actual_history={actual_history}, "
              f"actual_horizon={actual_horizon}, max_path_length={max_path_length}, "
              f"num_inference_steps={num_inference_steps}")

        assert num_inference_steps > 0, "num_inference_steps must be greater than 0"
        
        if not self.has_local_query:
            local_query = None
        if not self.has_global_query:
            global_query = None

        cond = get_conditions(start_states)
        
        # Track per-step losses for detailed logging
        step_losses = []
        
        # Check if we have full trajectories for ground truth comparison
        has_gt_trajectories = full_trajectories is not None and len(full_trajectories) > 0

        for step in range(num_inference_steps):
            sol = self.conditional_sample(cond, (batch_size, self.prediction_length, self.input_dim), global_query=global_query, local_query=local_query, return_chain=False, **sample_kwargs)

            final_states = sol.trajectories[:, -self.history_length:, :] 
            
            # Compute intermediate loss at this step (distance to target)
            intermediate_final = sol.trajectories[:, -1, :]
            step_loss, step_info = self.loss_fn(intermediate_final, target_final_states)
            
            # Compute ground truth comparison if available
            gt_loss = None
            gt_info = {}
            if has_gt_trajectories:
                # Calculate the timestep index for this rollout step
                # After step k, we've covered: actual_history + k * actual_horizon timesteps
                current_timestep = actual_history + (step + 1) * actual_horizon - 1
                
                # Get ground truth states at this timestep for each trajectory
                gt_states = []
                for i, traj in enumerate(full_trajectories[:batch_size]):
                    # Clamp to trajectory length
                    t_idx = min(current_timestep, len(traj) - 1)
                    gt_states.append(traj[t_idx])
                
                gt_states = torch.stack(gt_states).to(intermediate_final.device)
                gt_loss, gt_info = self.loss_fn(intermediate_final, gt_states)
            
            step_losses.append({
                'step': step + 1,
                'loss': step_loss.item(),
                'x_loss': step_info.get('x_loss', torch.tensor(0.0)).item() if hasattr(step_info.get('x_loss', 0), 'item') else step_info.get('x_loss', 0),
                'theta_loss': step_info.get('theta_loss', torch.tensor(0.0)).item() if hasattr(step_info.get('theta_loss', 0), 'item') else step_info.get('theta_loss', 0),
                'x_dot_loss': step_info.get('x_dot_loss', torch.tensor(0.0)).item() if hasattr(step_info.get('x_dot_loss', 0), 'item') else step_info.get('x_dot_loss', 0),
                'theta_dot_loss': step_info.get('theta_dot_loss', torch.tensor(0.0)).item() if hasattr(step_info.get('theta_dot_loss', 0), 'item') else step_info.get('theta_dot_loss', 0),
                'gt_loss': gt_loss.item() if gt_loss is not None else None,
            })
            
            # Print step-by-step loss (vs target)
            print(f"  [Rollout Step {step+1}/{num_inference_steps}] "
                  f"loss_vs_target={_fmt(step_loss)} | "
                  f"x={_fmt(step_info.get('x_loss', 0))} | "
                  f"θ={_fmt(step_info.get('theta_loss', 0))} | "
                  f"ẋ={_fmt(step_info.get('x_dot_loss', 0))} | "
                  f"θ̇={_fmt(step_info.get('theta_dot_loss', 0))}")
            
            # Print ground truth comparison if available
            if gt_loss is not None:
                current_timestep = actual_history + (step + 1) * actual_horizon - 1
                print(f"    [vs Ground Truth @ t={current_timestep}] "
                      f"loss_vs_gt={_fmt(gt_loss)} | "
                      f"x={_fmt(gt_info.get('x_loss', 0))} | "
                      f"θ={_fmt(gt_info.get('theta_loss', 0))} | "
                      f"ẋ={_fmt(gt_info.get('x_dot_loss', 0))} | "
                      f"θ̇={_fmt(gt_info.get('theta_dot_loss', 0))}")

            cond = get_conditions(final_states) # Start states for the next horizon

        pred_final_states = sol.trajectories[:, -1, :] 

        final_state_loss, final_state_info = self.loss_fn(pred_final_states, target_final_states)
        
        # Add step losses to info for potential analysis
        final_state_info['step_losses'] = step_losses

        return final_state_loss, final_state_info

    def evaluate_sequential_validation(
        self,
        trajectories: list,
        get_conditions: Callable,
        max_trajectories: int = None,
        verbose: bool = False,
        autoregressive_results: dict = None,
        **sample_kwargs
    ):
        """
        Evaluate model performance on consecutive segments from start to end of trajectories.
        Uses ground-truth conditions (same as validation_loss) but processes segments sequentially.
        
        Args:
            trajectories: List of full trajectories (torch.Tensor, each shape: [traj_length, dim])
            get_conditions: Function to convert history to conditions dict
            max_trajectories: Limit number of trajectories to evaluate
            verbose: Whether to print detailed output
            autoregressive_results: Results from final state evaluation for comparison
            **sample_kwargs: Additional arguments for conditional_sample
            
        Returns:
            dict with aggregated metrics and per-segment details
        """
        from genMoPlan.datasets.utils import apply_padding
        
        if max_trajectories is not None:
            trajectories = trajectories[:max_trajectories]
        
        all_segment_details = []
        trajectory_summaries = []
        
        actual_history = compute_actual_length(self.history_length, self.stride)
        horizon_length = self.prediction_length - self.history_length
        actual_horizon = compute_actual_length(horizon_length, self.stride)
        
        # Check if this is single-step inference mode (horizon covers entire trajectory)
        # Get the trajectory length from the first trajectory
        if trajectories:
            first_traj = trajectories[0]
            if isinstance(first_traj, torch.Tensor):
                sample_traj_len = first_traj.shape[0]
            elif hasattr(first_traj, '__len__'):
                sample_traj_len = len(first_traj)
            else:
                sample_traj_len = 0
            if verbose:
                print(f"    [DEBUG] first_traj type: {type(first_traj)}, shape: {first_traj.shape if hasattr(first_traj, 'shape') else 'N/A'}")
        else:
            sample_traj_len = 0
            
        is_single_step_mode = actual_horizon >= sample_traj_len - actual_history
        
        if verbose and is_single_step_mode:
            print(f"    [Single-step mode] actual_horizon ({actual_horizon}) >= trajectory_length ({sample_traj_len})")
            print(f"    Evaluating single padded segment per trajectory")
        
        # Prepare autoregressive comparison data if available
        autoregressive_step_losses = None
        autoregressive_avg_loss = None
        
        if autoregressive_results is not None:
            # Get average loss from autoregressive evaluation
            if 'final_state_loss' in autoregressive_results:
                autoregressive_avg_loss = autoregressive_results['final_state_loss']
            
            # Extract step losses from autoregressive evaluation
            # step_losses is a list of dicts, one per step per trajectory
            autoregressive_step_losses = {}
            if 'step_losses' in autoregressive_results:
                step_losses_list = autoregressive_results['step_losses']
                # Group by step number and average across trajectories
                for step_data in step_losses_list:
                    if isinstance(step_data, dict):
                        step = step_data.get('step', 0)
                        step_loss = step_data.get('loss', 0)
                        if step not in autoregressive_step_losses:
                            autoregressive_step_losses[step] = []
                        autoregressive_step_losses[step].append(step_loss)
                # Average across trajectories for each step
                for step in autoregressive_step_losses:
                    autoregressive_step_losses[step] = np.mean(autoregressive_step_losses[step])
        
        for traj_idx, trajectory in enumerate(trajectories):
            if not isinstance(trajectory, torch.Tensor):
                trajectory = torch.tensor(trajectory, dtype=torch.float32)
            
            # Move trajectory to same device as model
            device = next(self.parameters()).device
            trajectory = trajectory.to(device)
            
            trajectory_segments = []
            segment_idx = 0
            current_timestep = 0
            
            # Process consecutive segments until end of trajectory
            # Use relaxed condition to allow at least one segment even for single-step inference
            # where actual_horizon >= trajectory length (with padding)
            while current_timestep + actual_history <= len(trajectory):
                # Check if we have at least some horizon to predict (even if needs padding)
                if current_timestep + actual_history >= len(trajectory):
                    break  # No room for any horizon
                
                # Extract ground-truth history at current position
                history_start = current_timestep
                history_end = history_start + actual_history
                gt_history = trajectory[history_start:history_end:self.stride]
                
                # Extract ground-truth horizon following history
                horizon_start = history_end
                horizon_end = min(horizon_start + actual_horizon, len(trajectory))
                gt_horizon = trajectory[horizon_start:horizon_end:self.stride]
                
                # Skip if horizon is empty (shouldn't happen with above check, but safety)
                if len(gt_horizon) == 0:
                    break
                
                # Apply padding if needed (same as current validation)
                if len(gt_history) < self.history_length:
                    gt_history = apply_padding(gt_history, self.history_length, pad_left=True)
                if len(gt_horizon) < horizon_length:
                    pad_value = gt_horizon[-1] if len(gt_horizon) > 0 else gt_history[-1]
                    gt_horizon = apply_padding(gt_horizon, horizon_length, pad_left=False, pad_value=pad_value)
                
                # Get condition from ground-truth history (SAME as current validation)
                cond = get_conditions(gt_history)
                
                # Create full trajectory tensor for validation_loss
                gt_full_trajectory = torch.cat([gt_history, gt_horizon], dim=0)  # Shape: (prediction_length, dim)
                gt_full_trajectory = gt_full_trajectory.unsqueeze(0)  # Add batch dim: (1, prediction_length, dim)
                
                # Use same validation_loss method as current validation
                loss, info = self.validation_loss(
                    gt_full_trajectory,  # x: ground-truth segment
                    cond,                # condition from ground-truth history
                    verbose=False,
                    **sample_kwargs
                )
                
                # Calculate position in trajectory (normalized 0.0 to 1.0)
                position = current_timestep / len(trajectory) if len(trajectory) > 0 else 0.0
                
                # Match with autoregressive results if available
                autoregressive_loss_at_timestep = None
                autoregressive_vs_sequential_ratio = None
                
                if autoregressive_step_losses is not None:
                    # Calculate which autoregressive step corresponds to this timestep
                    # Autoregressive step k (1-indexed) reaches: actual_history + k * actual_horizon - 1
                    # Sequential segment ends at: horizon_end - 1
                    # Find closest autoregressive step
                    target_timestep = horizon_end - 1
                    best_step = None
                    min_diff = float('inf')
                    
                    for step in sorted(autoregressive_step_losses.keys()):
                        # step is 1-indexed in autoregressive results
                        ar_timestep = actual_history + step * actual_horizon - 1
                        diff = abs(ar_timestep - target_timestep)
                        if diff < min_diff:
                            min_diff = diff
                            best_step = step
                    
                    if best_step is not None and min_diff < actual_horizon:  # Only match if reasonably close
                        autoregressive_loss_at_timestep = autoregressive_step_losses[best_step]
                        # Handle negative or zero losses (can happen with some loss functions or numerical issues)
                        if abs(loss.item()) > 1e-8:  # Avoid division by zero or near-zero
                            autoregressive_vs_sequential_ratio = autoregressive_loss_at_timestep / abs(loss.item())
                        else:
                            # If sequential loss is essentially zero, ratio is undefined
                            autoregressive_vs_sequential_ratio = None
                
                # Track segment details - convert all tensor values to Python floats
                def to_float(val):
                    if val is None:
                        return 0.0
                    if isinstance(val, (int, float)):
                        return float(val)
                    if hasattr(val, 'item'):
                        return val.item()
                    return float(val)
                
                segment_detail = {
                    'trajectory_idx': traj_idx,
                    'segment_idx': segment_idx,
                    'history_start_timestep': history_start,
                    'history_end_timestep': history_end,
                    'horizon_start_timestep': horizon_start,
                    'horizon_end_timestep': horizon_end,
                    'position_in_trajectory': position,
                    'segment_loss': to_float(loss),
                    'final_state_loss': to_float(info.get('final_state_loss', 0)),
                    'x_loss': to_float(info.get('x_loss', 0)),
                    'theta_loss': to_float(info.get('theta_loss', 0)),
                    'x_dot_loss': to_float(info.get('x_dot_loss', 0)),
                    'theta_dot_loss': to_float(info.get('theta_dot_loss', 0)),
                }
                
                if autoregressive_loss_at_timestep is not None:
                    segment_detail['autoregressive_loss_at_same_timestep'] = autoregressive_loss_at_timestep
                    segment_detail['autoregressive_vs_sequential_ratio'] = autoregressive_vs_sequential_ratio
                
                all_segment_details.append(segment_detail)
                trajectory_segments.append(segment_detail)
                
                # Move to next consecutive segment
                current_timestep += actual_horizon  # Non-overlapping consecutive segments
                segment_idx += 1
            
            # Trajectory summary
            if trajectory_segments:
                segment_losses = [s['segment_loss'] for s in trajectory_segments]
                trajectory_summaries.append({
                    'trajectory_idx': traj_idx,
                    'trajectory_length': len(trajectory),
                    'num_segments': len(trajectory_segments),
                    'avg_loss': np.mean(segment_losses),
                    'first_segment_loss': trajectory_segments[0]['segment_loss'],
                    'last_segment_loss': trajectory_segments[-1]['segment_loss'],
                    'loss_trend': trajectory_segments[-1]['segment_loss'] - trajectory_segments[0]['segment_loss'],
                })
        
        # Aggregate metrics
        if not all_segment_details:
            return {}
        
        # Calculate aggregate metrics
        avg_segment_loss = np.mean([s['segment_loss'] for s in all_segment_details])
        avg_final_state_loss = np.mean([s['final_state_loss'] for s in all_segment_details])
        
        # Position-based analysis
        positions = [s['position_in_trajectory'] for s in all_segment_details]
        losses_by_position = [s['segment_loss'] for s in all_segment_details]
        
        early_segments = [s for s in all_segment_details if s['position_in_trajectory'] < 0.25]
        middle_segments = [s for s in all_segment_details if 0.25 <= s['position_in_trajectory'] < 0.75]
        late_segments = [s for s in all_segment_details if s['position_in_trajectory'] >= 0.75]
        
        early_segments_loss = np.mean([s['segment_loss'] for s in early_segments]) if early_segments else 0
        middle_segments_loss = np.mean([s['segment_loss'] for s in middle_segments]) if middle_segments else 0
        late_segments_loss = np.mean([s['segment_loss'] for s in late_segments]) if late_segments else 0
        
        # Per-component averages
        avg_x_loss = np.mean([s['x_loss'] for s in all_segment_details])
        avg_theta_loss = np.mean([s['theta_loss'] for s in all_segment_details])
        avg_x_dot_loss = np.mean([s['x_dot_loss'] for s in all_segment_details])
        avg_theta_dot_loss = np.mean([s['theta_dot_loss'] for s in all_segment_details])
        
        # Autoregressive comparison
        autoregressive_avg_loss = None
        autoregressive_vs_sequential_ratio = None
        error_accumulation_factor = None
        
        if autoregressive_results is not None:
            # Get average loss from autoregressive evaluation
            if 'final_state_loss' in autoregressive_results:
                autoregressive_avg_loss = autoregressive_results['final_state_loss']
                avg_segment_loss_abs = abs(avg_segment_loss)  # Handle negative losses
                if avg_segment_loss_abs > 1e-8:  # Avoid division by zero
                    autoregressive_vs_sequential_ratio = autoregressive_avg_loss / avg_segment_loss_abs
                    error_accumulation_factor = autoregressive_vs_sequential_ratio
                else:
                    autoregressive_vs_sequential_ratio = None
                    error_accumulation_factor = None
        
        # Build result dictionary
        result = {
            'avg_segment_loss': avg_segment_loss,
            'avg_final_state_loss': avg_final_state_loss,
            'num_segments': len(all_segment_details),
            'num_trajectories': len(trajectory_summaries),
            'segments_per_trajectory': len(all_segment_details) / len(trajectory_summaries) if trajectory_summaries else 0,
            'early_segments_loss': early_segments_loss,
            'middle_segments_loss': middle_segments_loss,
            'late_segments_loss': late_segments_loss,
            'avg_x_loss': avg_x_loss,
            'avg_theta_loss': avg_theta_loss,
            'avg_x_dot_loss': avg_x_dot_loss,
            'avg_theta_dot_loss': avg_theta_dot_loss,
            'segment_details': all_segment_details,
            'trajectory_summaries': trajectory_summaries,
        }
        
        if autoregressive_avg_loss is not None:
            result['autoregressive_avg_loss'] = autoregressive_avg_loss
            result['autoregressive_vs_sequential_ratio'] = autoregressive_vs_sequential_ratio
            result['error_accumulation_factor'] = error_accumulation_factor
        
        return result

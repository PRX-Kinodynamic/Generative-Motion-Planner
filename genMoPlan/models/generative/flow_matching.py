import warnings
import numpy as np
from typing import Callable

import torch

try:
    from flow_matching.path.scheduler import *
    from flow_matching.path import *
    from flow_matching.solver import *

    FLOW_MATCHING_AVAILABLE = True
except ImportError:
    FLOW_MATCHING_AVAILABLE = False


from genMoPlan.models.helpers import apply_conditioning
from genMoPlan.utils.arrays import torch_randn_like
from genMoPlan.datasets.constants import MASK_ON, MASK_OFF

from .base import GenerativeModel, Sample


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
        # Flow matching specific parameters
        scheduler="CondOTScheduler",
        path="AffineProbPath",
        solver="ODESolver",
        n_fourier_features=1,
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
            **kwargs
        )

        self.n_timesteps = 1
        self.history_length = history_length

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

    def compute_loss(self, x_target, cond, global_query=None, local_query=None, seed=None, mask=None):
        """
        Choose a random timestep t and calculate the loss for the model.

        When use_mask_loss_weighting is True, masked positions (MASK_ON=0) will
        have zero loss contribution, focusing training on valid positions only.
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

        # Compute combined loss weights (combining temporal weights and mask weights)
        combined_loss_weights = self.loss_weights

        if self.use_mask_loss_weighting and mask is not None:
            # Create mask weights: MASK_OFF (1) = count in loss, MASK_ON (0) = ignore
            # Expand mask from [batch, seq] to [batch, seq, 1] for broadcasting
            mask_weights = mask.unsqueeze(-1).to(dtype=x_target.dtype, device=x_target.device)

            if combined_loss_weights is not None:
                # Combine with existing loss weights
                combined_loss_weights = combined_loss_weights * mask_weights
            else:
                combined_loss_weights = mask_weights

        loss, info = self.loss_fn(
            self.vector_field(x=path_sample.x_t, t=path_sample.t, global_query=global_query, local_query=local_query, mask=mask),
            path_sample.dx_t,
            ignore_manifold=True,
            loss_weights=combined_loss_weights,
        )

        # Add mask statistics to info
        if self.use_mask_loss_weighting and mask is not None:
            num_masked = (mask == MASK_ON).sum().item()
            num_valid = (mask == MASK_OFF).sum().item()
            info["num_masked_positions"] = num_masked
            info["num_valid_positions"] = num_valid
            info["percent_masked"] = 100.0 * num_masked / mask.numel() if mask.numel() > 0 else 0.0

        return loss, info

    
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

    def validation_loss(self, x, cond, global_query=None, local_query=None, mask=None, **sample_kwargs):
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

        final_state_loss, _ = self.loss_fn(pred_final_state, target_final_state)

        loss, info = self.loss_fn(sol.trajectories, x, loss_weights=self.loss_weights)

        info["final_state_loss"] = final_state_loss

        return loss, info

    def evaluate_final_states(self, start_states: torch.Tensor, target_final_states: torch.Tensor, global_query: torch.Tensor =None, local_query: torch.Tensor =None, max_path_length: int =None, get_conditions: Callable =None, **sample_kwargs):
        """
        Evaluate the final states of the model given the start states and the target final states.

        Args:
            start_states: The start states to evaluate the final states from. (batch_size, history_length, dim)
            target_final_states: The target final states to evaluate the final states from. (batch_size, dim)
            global_query: The global query to use for the model.
            local_query: The local query to use for the model.
            max_path_length: The maximum path length to use for the model.
            get_conditions: The function to get the conditions for the model.
            **sample_kwargs: Additional arguments passed to the conditional sample method.

        Returns:
            final_state_loss: The loss for the final states.
            final_state_info: The information for the final states.
        """

        assert max_path_length is not None, "max_path_length must be provided for final state evaluation"
        assert get_conditions is not None, "get_conditions must be provided for final state evaluation"

        batch_size = target_final_states.shape[0]

        num_inference_steps = np.ceil((max_path_length - self.history_length) / self.prediction_length).astype(int)

        assert num_inference_steps > 0, "num_inference_steps must be greater than 0"
        
        if not self.has_local_query:
            local_query = None
        if not self.has_global_query:
            global_query = None

        cond = get_conditions(start_states)

        for _ in range(num_inference_steps):
            sol = self.conditional_sample(cond, (batch_size, self.prediction_length, self.input_dim), global_query=global_query, local_query=local_query, return_chain=False, **sample_kwargs)

            final_states = sol.trajectories[:, -self.history_length:, :] 

            cond = get_conditions(final_states) # Start states for the next horizon

        pred_final_states = sol.trajectories[:, -1, :] 

        final_state_loss, final_state_info = self.loss_fn(pred_final_states, target_final_states)

        return final_state_loss, final_state_info


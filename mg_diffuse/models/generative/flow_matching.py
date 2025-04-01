from collections import namedtuple
import warnings

from torch import nn
import torch
import numpy as np

from mg_diffuse.models.helpers import (
    apply_conditioning,
)

from flow_matching.path.scheduler import *
from flow_matching.path import *
from flow_matching.solver import *

from .abs_gen_model import GenerativeModel, Sample


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
        loss_weights=None,
        loss_discount=1.0,
        action_indices=None,
        has_query=False,
        # Flow matching specific parameters
        scheduler="CondOTScheduler",
        path="CondOTProbPath",
        solver="ODESolver",
        **kwargs,
    ):
        super().__init__(
            model=model,
            input_dim=input_dim,
            output_dim=output_dim,
            prediction_length=prediction_length,
            history_length=history_length,
            clip_denoised=clip_denoised,
            loss_type=loss_type,
            loss_weights=loss_weights,
            loss_discount=loss_discount,
            action_indices=action_indices,
            has_query=has_query,
            **kwargs
        )
        
        self.n_timesteps = 1
        self.history_length = history_length
        
        # ------------------------------ Setup flow matching components ------------------------------#

        path_args = []
        if scheduler is not None:
            if type(scheduler) == str:
                scheduler = eval(scheduler)
            path_args.append(scheduler())

        if type(path) == str:
            path = eval(path)
        self.path = path(*path_args)

        if type(solver) == str:
            solver = eval(solver)
        self.solver = solver(self.vector_field)

    # --------------------------------------- vector field ----------------------------------------#

    def vector_field(self, x=None, t=None, query=None):
        if x is None or t is None:
            raise ValueError("x and t must be provided")
        
        if t.ndim == 0:
            batch_size = x.shape[0]
            t = t.unsqueeze(0).repeat(batch_size)

        # The model expects parameters in the order: (x, query, t)
        return self.model(x, query, t)
    
    # ------------------------------------------ training ------------------------------------------#

    def compute_loss(self, x_target, cond, query=None):
        """
        Choose a random timestep t and calculate the loss for the model
        """
        batch_size = len(x_target)
        t = torch.rand(batch_size, device=x_target.device)

        x_noisy = torch.randn_like(x_target)
        x_noisy = apply_conditioning(x_noisy, cond)

        path_sample = self.path.sample(t=t, x_0=x_noisy, x_1=x_target)

        loss, info = self.loss_fn(self.vector_field(x=path_sample.x_t, t=path_sample.t), path_sample.dx_t, loss_weights=self.loss_weights)

        return loss, info

    
    # ------------------------------------------ inference ------------------------------------------#

    @torch.no_grad()
    def conditional_sample(self, cond, shape, query=None, step_size=0.05,  integration_method="midpoint", return_chain=False, n_intermediate_steps=0, **kwargs):
        """
        Generate samples by running the flow matching ODE solver from noise to target.
        
        Args:
            cond: Conditioning information that will be applied to the samples
            shape: Shape of the output samples
            query: Optional query tensor for conditional generation
            step_size: Step size for the ODE solver
            integration_method: Integration method to use ("midpoint", "euler", etc.)
            return_chain: Whether to return intermediate states in the sampling chain
            n_intermediate_steps: Number of intermediate steps to save (if return_chain=True)
            **kwargs: Additional arguments passed to the solver
            
        Returns:
            Sample: An object containing the generated trajectories and optional intermediate chains
        """

        device = self.loss_weights.device

        assert n_intermediate_steps >= 0

        if n_intermediate_steps == 0 and return_chain:
            warnings.warn("n_intermediate_steps is 0 and return_chain is True, this will return only the noisy and final samples")

        T = torch.linspace(0, 1, n_intermediate_steps + 2)

        x_noisy = torch.randn(shape, device=device)
        x_noisy = apply_conditioning(x_noisy, cond)

        # Make query explicit in model_extras
        model_extras = {}
        if query is not None:
            model_extras['query'] = query

        sol = self.solver.sample(
            x_init=x_noisy, 
            step_size=step_size, 
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

        return Sample(trajectories=sol, values=None, chains=chains)
    
    # ------------------------------------------ validation ------------------------------------------#

    def validation_loss(self, x, cond, query, **sample_kwargs):
        if not self.has_query:
            query = None

        sol = self.conditional_sample(cond, x.shape, query=query, verbose=False, return_chain=False, **sample_kwargs)

        loss, info = self.loss_fn(sol.trajectories, x, loss_weights=self.loss_weights)

        return loss, info

from collections import namedtuple

from torch import nn
import torch

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from .helpers import apply_conditioning
from .helpers.losses import Losses


Sample = namedtuple("Sample", "trajectories values chains")

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        t=t.unsqueeze(0)  # for the model t has to have dimension but the ODE solver gives as input
        return self.model(x, t)


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    # t_int = (self.n_timesteps*t).long()
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class FlowMatching(nn.Module):
    def __init__(
        self,
        model,
        observation_dim,
        horizon,
        n_timesteps=100,
        clip_denoised=False,
        predict_epsilon=True,
        loss_type="l1",
        loss_weights=None,
        loss_discount=1.0,
    ):
        super().__init__()

        self.model = model

        self.observation_dim = observation_dim
        self.transition_dim = observation_dim
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.n_timesteps = int(n_timesteps)
        self.horizon = horizon

        loss_weights = self.get_loss_weights(loss_discount, loss_weights)

        self.loss_fn = Losses[loss_type](loss_weights)

        # instantiate an affine path object
        self.prob_path = AffineProbPath(scheduler=CondOTScheduler())

        wrapped_vf = WrappedModel(self.model)

        self.solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class

    def get_loss_weights(self, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        return loss_weights

    @torch.no_grad()
    def flow(
        self,
        shape,
        cond,
        verbose=True,
        return_chain=False,
        **sample_kwargs
    ):
        """

        Apply conditioning to x by fixing the states in x at the given timesteps from cond

        Then loop through the timesteps in reverse order and sample from the model, applying conditioning at each step
        """

        # device = self.device
        batch_size = shape[0]

        # step size for ode solver
        step_size = 0.05
        eps_time = 1e-2
        T = torch.linspace(0,1,self.n_timesteps)  # sample times
        # T = T.to(device=device)

        # x_init = torch.randn(shape, device=device)
        x_init = torch.randn(shape)
        x_init = apply_conditioning(x_init, cond)

        # sample from the model
        sol = self.solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)  

        x_end = sol[-1]

        chain = sol if return_chain else None

        values = torch.zeros(len(x_end), device=x_end.device)

        x_end, values = sort_by_values(x_end, values)

        return Sample(x_end, values, chain)


    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        """
        conditions : [ (time, state), ... ]
        """
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.flow(shape, cond, **sample_kwargs)

    # ------------------------------------------ training ------------------------------------------#
    
    def loss_vf(self, x, cond):
        # Get a normal distribution of noise and sample a noisy x by adding a scaled noise to a scaled x_start
        # Apply conditioning to the noisy x
        noise = torch.randn_like(x)
        batch_size = len(x)
        t = torch.rand((batch_size,), device=x.device)
        # t = torch.rand(x.shape[0], device=x.device)

        # x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        x_noisy = noise

        x_noisy = apply_conditioning(x_noisy, cond)

        # Apply conditioning to the target x
        x_target = apply_conditioning(x, cond)

        assert noise.shape == x_target.shape        

        path_sample = self.prob_path.sample(t=t, x_0=x_noisy, x_1=x_target)

        if self.predict_epsilon:
            raise NotImplementedError
        else:
            loss, info = self.loss_fn(self.model(path_sample.x_t, path_sample.t), path_sample.dx_t)

        return loss, info
    
    def loss(self, x, *args):
        """
        Calculate the loss for the model vector field
        """
        return self.loss_vf(x, *args)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

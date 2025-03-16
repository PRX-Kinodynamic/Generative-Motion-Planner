from collections import namedtuple

from torch import nn
import torch
import numpy as np

import mg_diffuse.utils as utils

from .helpers import (
    apply_conditioning,
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    sort_by_values,
)
from .helpers.losses import Losses


Sample = namedtuple("Sample", "trajectories values chains")


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    """
    Get the model_mean and the fixed variance from the model

    then sample noise from a normal distribution
    """
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
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

        # ----- calculations for diffusion noising and denoising parameters ------

        betas = cosine_beta_schedule(n_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior p(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

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

    def predict_start_from_noise(self, x_t, t, model_output):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly


        So if predicting the direct value, model_output is returned
        else, model_output is treated as noise and is then subtracted from x_t after scaling
        """
        if self.predict_epsilon:
            noise = model_output
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return model_output

    def q_posterior(self, x_start, x_t, t):
        """
        Get a mean of the x_start and x_t by using the posterior mean coefficients

        Return the mean along with the fixed variance and clipped log variance
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        """
        Reconstructs x by getting an output from the model and then either subtracting the noise and returning the output directly as x_con

        Clips the reconstructed x

        Then gets the mean, variance and log variance of the posterior distribution and returns them
        """

        x_recon = self.predict_start_from_noise(
            x, t=t, model_output=self.model(x, cond, t)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        verbose=True,
        return_chain=False,
        sample_fn=default_sample_fn,
        **sample_kwargs
    ):
        """

        Apply conditioning to x by fixing the states in x at the given timesteps from cond

        Then loop through the timesteps in reverse order and sample from the model, applying conditioning at each step
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond)

            progress.update(
                {"t": i, "vmin": values.min().item(), "vmax": values.max().item()}
            )
            if return_chain:
                chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        """
        conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        """
        Get a normal distribution of noise and sample a noisy x by adding a scaled noise to a scaled x_start
        Apply conditioning to the noisy x

        Then get the reconstructed x by passing the noisy x through the model and apply conditioning to the reconstructed x

        If predict epsilon, calculate the loss between the reconstructed x and the noise
        else, calculate the loss between the reconstructed x and the x_start
        """
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        """
        Choose a random timestep t and calculate the loss for the model
        """
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

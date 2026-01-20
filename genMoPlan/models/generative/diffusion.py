from collections import namedtuple

from torch import nn
import torch
import numpy as np

import genMoPlan.utils as utils
from genMoPlan.utils.arrays import torch_randn_like

from ..helpers import (
    apply_conditioning,
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    sort_by_values,
)

from .base import GenerativeModel, Sample


@torch.no_grad()
def default_sample_fn(model, x, global_query, local_query, t, generator=None, **kwargs):
    """
    Get the model_mean and the fixed variance from the model

    then sample noise from a normal distribution
    """
    # Thread through optional extras like masks to the model step
    model_mean, _, model_log_variance = model.p_mean_variance(
        x=x, global_query=global_query, local_query=local_query, t=t, **kwargs
    )
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch_randn_like(x, generator=generator)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class Diffusion(GenerativeModel):
    def __init__(
        self,
        model,
        system,  # System instance - REQUIRED, provides system-specific config
        # Model-specific params
        prediction_length=None,
        history_length=None,
        clip_denoised=False,
        loss_type="l2",
        has_local_query=False,
        has_global_query=False,
        # Diffusion specific parameters
        sort_by_values=False,
        n_timesteps=100,
        predict_epsilon=True,
        use_history_mask: bool = False,
        # Deprecated parameters (for backward compatibility - will be ignored)
        input_dim=None,
        output_dim=None,
        action_indices=None,
        manifold=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            system=system,
            prediction_length=prediction_length,
            history_length=history_length,
            clip_denoised=clip_denoised,
            loss_type=loss_type,
            has_local_query=has_local_query,
            has_global_query=has_global_query,
            use_history_mask=use_history_mask,
            **kwargs
        )
        
        self.sort_by_values = sort_by_values
        self.predict_epsilon = predict_epsilon
        self.n_timesteps = int(n_timesteps)
        # Explicitly disallow mask usage with classic diffusion
        if use_history_mask:
            raise NotImplementedError("use_history_mask is not supported for Diffusion; use Flow Matching instead")

        if self.manifold is not None:
            raise NotImplementedError("Manifold is not implemented yet for diffusion model")
        
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



    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_target, t, noise=None, generator=None):
        if noise is None:
            noise = torch_randn_like(x_target, generator=generator)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_target.shape) * x_target
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_target.shape) * noise
        )

        return sample

    def compute_loss(self, x_target, cond, global_query=None, local_query=None, seed=None, mask=None):
        """
        Get a normal distribution of noise and sample a noisy x by adding a scaled noise to a scaled x_start
        Apply conditioning to the noisy x

        Then get the reconstructed x by passing the noisy x through the model and apply conditioning to the reconstructed x

        If predict epsilon, calculate the loss between the reconstructed x and the noise
        else, calculate the loss between the reconstructed x and the x_start
        """
        if seed is not None:
            generator = torch.Generator(device=x_target.device).manual_seed(seed)
        else:
            generator = None

        if getattr(self, "use_mask", False) and mask is None:
            raise ValueError("Mask expected but not provided")

        batch_size = len(x_target)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_target.device, generator=generator).long()

        noise = torch_randn_like(x_target, generator=generator)

        x_noisy = self.q_sample(x_target=x_target, t=t, noise=noise, generator=generator)
        
        apply_conditioning(x_noisy, cond)

        # Only pass mask if provided and supported
        if mask is not None:
            x_recon = self.model(x_noisy, global_query, local_query, t, mask=mask)
        else:
            x_recon = self.model(x_noisy, global_query, local_query, t)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            apply_conditioning(x_recon, cond)
            loss, info = self.loss_fn(x_recon, x_target)

        return loss, info
    
    # ------------------------------------------ inference ------------------------------------------#

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
    
    def p_mean_variance(self, x, global_query, local_query, t, mask=None, **kwargs):
        """
        Reconstructs x by getting an output from the model and then either subtracting the noise and returning the output directly as x_con

        Clips the reconstructed x

        Then gets the mean, variance and log variance of the posterior distribution and returns them
        """

        # Pass mask to temporal model if provided and supported; otherwise fall back
        if mask is None:
            model_output = self.model(x, global_query=global_query, local_query=local_query, time=t)
        else:
            try:
                model_output = self.model(x, global_query=global_query, local_query=local_query, time=t, mask=mask)
            except TypeError:
                model_output = self.model(x, global_query=global_query, local_query=local_query, time=t)

        x_recon = self.predict_start_from_noise(x, t=t, model_output=model_output)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def conditional_sample(
        self, 
        cond, 
        shape, 
        global_query=None, 
        local_query=None, 
        verbose=False, 
        return_chain=False, 
        sample_fn=default_sample_fn, 
        seed=None,
        **sample_kwargs,
    ) -> Sample:
        """

        Apply conditioning to x by fixing the states in x at the given timesteps from cond

        Then loop through the timesteps in reverse order and sample from the model, applying conditioning at each step
        """
        device = self.betas.device

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        batch_size = shape[0]
        x = torch.randn(shape, device=device, generator=generator)
        apply_conditioning(x, cond)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, global_query, local_query, t, generator=generator, **sample_kwargs)
            apply_conditioning(x, cond)

            progress.update(
                {"t": i, "vmin": values.min().item(), "vmax": values.max().item()}
            )
            if return_chain:
                chain.append(x)

        progress.stamp()

        if self.sort_by_values:
            x, values = sort_by_values(x, values)

        if return_chain:
            chain = torch.stack(chain, dim=1)

        return Sample(trajectories=x, values=values, chains=chain)

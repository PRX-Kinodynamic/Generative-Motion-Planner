from typing import Tuple
from torch import nn
import numpy as np
import torch

from genMoPlan.utils.arrays import to_torch
from .losses import *
from .nn_helpers import *


def extract(a, t, x_shape):
    """
    a : tensor [... x timestep ]
    t : tensor [ batch_size x ... ]
    x_shape : tuple

    Gets the value of a at t index from the last dimension of a
    and reshapes it to a tensor of shape (x_shape, 1, 1, ....) and same number dimensions as x_shape

    returns : tensor [ batch_size x (1, ) x (1, ) x ... ]
    """

    batch_size, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ

    Args:
        timesteps: Number of timesteps
        s: Parameter for the cosine schedule
        dtype: Data type of the output tensor

    Returns:
        Tensor of shape (timesteps,) containing the beta schedule
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(x: torch.Tensor, conditions: dict) -> None:
    """
    Apply conditioning to the tensor x

    Args:
        x: Tensor to apply conditioning to
        conditions: Dictionary of conditions to apply

    Returns:
        Tensor with conditioning applied
    """

    if conditions is None:
        return

    for t, val in conditions.items():
        x[:, t] = val.clone()

def sort_by_values(x: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values

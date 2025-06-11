from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Union
from torch import nn
import torch

from genMoPlan.models.temporal.base import TemporalModel

from ..helpers import (
    get_loss_weights,
    Losses
)

Sample = namedtuple("Sample", "trajectories values chains")

class GenerativeModel(nn.Module, ABC):
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
        has_global_query=False,
        has_local_query=False,
        manifold=None,
        **kwargs,
    ):
        super().__init__()

        self.model: TemporalModel = model

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.clip_denoised = clip_denoised
        self.prediction_length = prediction_length
        self.action_indices = action_indices
        self.has_global_query = has_global_query
        self.has_local_query = has_local_query
        self.register_buffer("loss_weights", get_loss_weights(output_dim, prediction_length, loss_discount, loss_weights))
        self.manifold = manifold
        self.loss_fn = Losses[loss_type](history_length, action_indices, manifold=manifold)

    @abstractmethod
    def compute_loss(self, x, *args):
        """
        Compute the loss for the model.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    @torch.no_grad()
    def conditional_sample(self, cond, **kwargs):
        """
        Generate samples conditioned on the input.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def loss(self, x, cond, global_query=None, local_query=None):
        if not self.has_local_query:
            local_query = None

        if not self.has_global_query:
            global_query = None

        return self.compute_loss(x, cond, global_query, local_query)

    @torch.no_grad()
    def forward(self, cond, global_query=None, local_query=None, verbose=True, return_chain=False, **kwargs):
        batch_size = len(cond[0])
        shape = (batch_size, self.prediction_length, self.output_dim)
        return self.conditional_sample(cond, shape, global_query=global_query, local_query=local_query, verbose=verbose, return_chain=return_chain, **kwargs) 
    

    @torch.no_grad()
    def validation_loss(self, x, cond, global_query=None, local_query=None, **kwargs):
        """
        Compute the validation loss for the model.
        """
        if not self.has_local_query:
            local_query = None

        if not self.has_global_query:
            global_query = None

        return self.compute_loss(x, cond, global_query, local_query)
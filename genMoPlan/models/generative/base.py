from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Union
from torch import nn
import torch

from genMoPlan.models.temporal.base import TemporalModel

from ..helpers import (
    Losses, 
    LossWeightTypes,
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
        action_indices=None,
        has_global_query=False,
        has_local_query=False,
        manifold=None,
        val_seed=42,
        state_names=None,
        loss_weight_type="none",
        loss_weight_kwargs={},
        use_history_mask: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model: TemporalModel = model

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.clip_denoised = clip_denoised
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.action_indices = action_indices
        self.has_global_query = has_global_query
        self.has_local_query = has_local_query
        self.manifold = manifold
        self.use_mask = bool(use_history_mask)
        if loss_weight_type is not None and loss_weight_type != "none":
            loss_weights = LossWeightTypes[loss_weight_type](history_length=history_length, prediction_length=prediction_length, **loss_weight_kwargs)
            self.register_buffer("loss_weights", loss_weights)
        else:
            self.loss_weights = None
        self.loss_fn = Losses[loss_type](history_length, action_indices, manifold=manifold, state_names=state_names)

        self.val_seed = val_seed

        # Enforce that history masking is only supported with the Diffusion Transformer temporal model
        if self.use_mask:
            from genMoPlan.models.temporal.diffusionTransformer import TemporalDiffusionTransformer
            if not isinstance(self.model, TemporalDiffusionTransformer):
                raise NotImplementedError("Masking requires TemporalDiffusionTransformer as the temporal model")

        # Propagate mask expectation to temporal model so it can assert on forward
        # Propagate mask expectation to the temporal model
        try:
            setattr(self.model, "expect_mask", self.use_mask)
        except Exception:
            pass

    @abstractmethod
    def compute_loss(self, x, *args):
        """
        Compute the loss for the model.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    @torch.no_grad()
    def conditional_sample(self, cond, **kwargs) -> Sample:
        """
        Generate samples conditioned on the input.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def loss(self, x, cond, global_query=None, local_query=None, mask=None):
        if not self.has_local_query:
            local_query = None

        if not self.has_global_query:
            global_query = None

        # Treat empty mask tensors as None
        if mask is not None and torch.is_tensor(mask) and mask.numel() == 0:
            mask = None

        return self.compute_loss(x, cond, global_query, local_query, mask=mask)

    @torch.no_grad()
    def forward(self, cond, global_query=None, local_query=None, verbose=True, return_chain=False, mask=None, **kwargs) -> Sample:
        if cond[0].ndim == 1:
            batch_size = 1
        else:
            batch_size = len(cond[0])

        cond = {t: cond[t].unsqueeze(0) for t in cond}

        if global_query is not None and global_query.ndim == 1:
            global_query = global_query.unsqueeze(0)
        if local_query is not None and local_query.ndim == 1:
            local_query = local_query.unsqueeze(0)
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)

        shape = (batch_size, self.prediction_length, self.output_dim)
        return self.conditional_sample(cond, shape, global_query=global_query, local_query=local_query, verbose=verbose, return_chain=return_chain, mask=mask, **kwargs) 
    

    @torch.no_grad()
    def validation_loss(self, x, cond, global_query=None, local_query=None, mask=None, **kwargs):
        """
        Compute the validation loss for the model.
        """

        if not self.has_local_query:
            local_query = None

        if not self.has_global_query:
            global_query = None

        # Treat empty mask tensors as None
        if mask is not None and torch.is_tensor(mask) and mask.numel() == 0:
            mask = None

        return self.compute_loss(x, cond, global_query, local_query, seed=self.val_seed, mask=mask)
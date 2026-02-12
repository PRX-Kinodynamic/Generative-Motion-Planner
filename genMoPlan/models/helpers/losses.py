from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import genMoPlan.utils as utils


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):
    def __init__(self, history_length=1, action_indices=None, manifold=None):
        super().__init__()
        self.history_length = history_length
        self.action_indices = action_indices
        self.manifold = manifold

    def forward(self, pred, targ, ignore_manifold=False, loss_weights=None):
        """
        pred, targ : tensor
            [ batch_size x horizon x output_dim ]
        """
        raw_loss = self._loss(pred, targ, ignore_manifold) # [ batch_size x horizon x output_dim ]

        if loss_weights is not None:
            weighted_loss = raw_loss * loss_weights
        else:
            weighted_loss = raw_loss

        mean_loss = weighted_loss.mean()

        info = {}
        info["raw_loss"] = raw_loss.mean()
        info["weighted_loss"] = weighted_loss.mean()

        return mean_loss, info


class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ, ignore_manifold=False):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(), utils.to_np(targ).squeeze()
            )[0, 1]
        else:
            corr = np.NaN

        info = {
            "mean_pred": pred.mean(),
            "mean_targ": targ.mean(),
            "min_pred": pred.min(),
            "min_targ": targ.min(),
            "max_pred": pred.max(),
            "max_targ": targ.max(),
            "corr": corr,
        }

        return loss, info


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ, ignore_manifold=False):
        # manifold is REQUIRED (always exists); use ignore_manifold to opt-out (e.g., dx_t loss)
        if ignore_manifold:
            return torch.abs(pred - targ)
        return self.manifold.dist(pred, targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ, ignore_manifold=False):
        # Output shape: [batch_size, horizon, output_dim], not performing any reduction over the output_dim
        # manifold is REQUIRED (always exists); use ignore_manifold to opt-out (e.g., dx_t loss)
        if ignore_manifold:
            return F.mse_loss(pred, targ, reduction="none")
        return self.manifold.dist(pred, targ)


class ValueL1(ValueLoss):
    def _loss(self, pred, targ, ignore_manifold=False):
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):
    def _loss(self, pred, targ, ignore_manifold=False):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
    "value_l1": ValueL1,
    "value_l2": ValueL2,
}


def generate_next_history_loss_weights(
    *,
    history_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    lambda_next_history: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    assert history_length is not None, "history_length is required"
    assert prediction_length is not None, "prediction_length is required"
    assert lambda_next_history is not None, "lambda_next_history is required"

    weights = torch.ones(1, prediction_length, 1)

    weights[:, -history_length:] = lambda_next_history

    return weights


LossWeightTypes = {
    "none": None,
    "next_history": generate_next_history_loss_weights,
}


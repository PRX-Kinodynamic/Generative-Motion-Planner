import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import genMoPlan.utils as utils


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):

    def __init__(self, history_length=1, action_indices=None, manifold=None, state_names=None):
        super().__init__()
        self.history_length = history_length
        self.action_indices = action_indices
        self.manifold = manifold
        self.state_names = state_names

    def forward(self, pred, targ, ignore_manifold=False):
        """
        pred, targ : tensor
            [ batch_size x horizon x output_dim ]
        """
        loss = self._loss(pred, targ, ignore_manifold) # [ batch_size x horizon x output_dim ]

        if loss.ndim == 3:
            statewise_loss = loss.mean(axis=(0, 1))
        else:
            statewise_loss = loss.mean(axis=0)

        info = {}

        if self.state_names is not None:
            for i, state_name in enumerate(self.state_names):
                info[f"{state_name}_loss"] = statewise_loss[i]

        mean_loss = statewise_loss.mean()

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
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ, ignore_manifold=False):
        # Output shape: [batch_size, horizon, output_dim], not performing any reduction over the output_dim
        if self.manifold is not None and not ignore_manifold:
            return self.manifold.dist(pred, targ)

        return F.mse_loss(pred, targ, reduction="none")


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

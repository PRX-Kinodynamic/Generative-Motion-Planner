import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import mg_diffuse.utils as utils


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):

    def __init__(self, weights, history_length=1, action_indices=None):
        super().__init__()
        self.register_buffer("weights", weights)
        self.history_length = history_length
        self.action_indices = action_indices
    def forward(self, pred, targ, loss_weights=None):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        weights = loss_weights if loss_weights is not None else self.weights

        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        cond_loss = (
            loss[:, :self.history_length] / weights[:self.history_length]
        ).mean()

        if self.action_indices is not None:
            action_loss = loss[:, :, self.action_indices].mean()
        else:
            action_loss = 0

        return weighted_loss, {"cond_loss": cond_loss, "action_loss": action_loss}


class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ):
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

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
    "value_l1": ValueL1,
    "value_l2": ValueL2,
}

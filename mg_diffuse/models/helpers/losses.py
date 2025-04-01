import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import mg_diffuse.utils as utils


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):

    def __init__(self, history_length=1, action_indices=None):
        super().__init__()
        self.history_length = history_length
        self.action_indices = action_indices

    def forward(self, pred, targ, loss_weights):
        """
        pred, targ : tensor
            [ batch_size x horizon x output_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * loss_weights).mean()
        info = {}

        if self.history_length > 1:
            info["cond_loss"] = (loss[:, :self.history_length] / loss_weights[:self.history_length]).mean()

        if self.action_indices is not None:
            info["action_loss"] = loss[:, :, self.action_indices].mean()

        return weighted_loss, info


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


def get_loss_weights(output_dim, prediction_length, discount, weights_dict):
    '''
        sets loss coefficients for trajectory

        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
    '''

    dim_weights = torch.ones(output_dim, dtype=torch.float32)

    ## set loss coefficients for dimensions of observation
    if weights_dict is None: weights_dict = {}
    for ind, w in weights_dict.items():
        dim_weights[ind] *= w

    ## decay loss with trajectory timestep: discount**t
    discounts = discount ** torch.arange(prediction_length, dtype=torch.float)
    discounts = discounts / discounts.mean()
    loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

    return loss_weights
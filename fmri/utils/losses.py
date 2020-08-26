# Class from https://github.com/ex4sperans/variational-inference-with-normalizing-flows/blob/master/flow.pyimport numpy as np
from torch import nn
import torch

def safe_log(z):
    import torch
    return torch.log(z + 1e-7)


class FreeEnergyBound(nn.Module):

    def __init__(self):
        super().__init__()
        self.density = p_z

    def forward(self, zk, log_jacobians):
        sum_of_log_jacobians = sum(log_jacobians)
        KLD = torch.mean(sum_of_log_jacobians - safe_log(self.density(zk)))
        return KLD

def QuantileLoss(preds, target, quantiles):
    def _tilted_loss(q, e):
        return torch.max((q-1) * e, q * e).unsqueeze(1)

    err = target - preds
    losses = [_tilted_loss(q, err[:, i])  # calculate per quantile
              for i, q in enumerate(quantiles)]

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss

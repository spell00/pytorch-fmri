# Class from https://github.com/ex4sperans/variational-inference-with-normalizing-flows/blob/master/flow.pyimport numpy as np
from torch import nn
import torch
from graveyard.densities import p_z

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

import torch.nn.functional as F
from random import random, randint
from tqdm import tqdm
import torch

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = tensor - torch.min(tensor)
        return tensor / torch.max(tensor)

    def __repr__(self):
        return self.__class__.__name__


class Flip90(object):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1:
            return tensor.transpose(1, 2)
        else:
            return tensor


class XFlip(object):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1:
            return tensor.flip(0)
        else:
            return tensor

class YFlip(object):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1:
            return tensor.flip(1)
        else:
            return tensor


class ZFlip(object):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1:
            return tensor.flip(2)
        else:
            return tensor


class Flip180(object):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1:
            return tensor.transpose(1, 2).flip(0)
        else:
            return tensor


class Flip270(object):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1:
            return tensor.transpose(1, 2).flip(2)
        else:
            return tensor


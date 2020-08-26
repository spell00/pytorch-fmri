import random
import argparse
import torch.nn as nn
import torch
from ..utils.masked_layer import GatedConv3d
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from medicaltorch import transforms as mt_transforms
from torch.distributions import Beta
from torch.distributions.gamma import Gamma

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"


def random_init(m, init_func=torch.nn.init.xavier_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


class GaussianSample(nn.Module):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features).to(device)
        self.log_var = nn.Linear(in_features, out_features).to(device)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return x, mu, log_var

    def mle(self, x):
        return self.mu(x)

class BetaSample(nn.Module):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(BetaSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a = nn.Linear(in_features, out_features).to(device)
        self.b = nn.Linear(in_features, out_features).to(device)

    def forward(self, x):
        a = self.a(x)
        b = F.softplus(self.b(x))
        pdf = torch._standard_gamma(a + b) * (x ** (a - 1)) * ((1 - x) ** (b - 1)) / (torch._standard_gamma(a) + torch._standard_gamma(b))

        return pdf


class ResBlock3D(nn.Module):
    def __init__(self, in_channel, channel, activation=nn.ReLU, device='cuda'):
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            activation(),
            nn.Conv3d(in_channel, channel, 3, padding=1),
            activation(),
            nn.Conv3d(channel, in_channel, 1),
        )
        self.conv.apply(random_init)

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class ResBlockDeconv3D(nn.Module):
    def __init__(self, in_channel, channel, activation=nn.ReLU, device='cuda'):
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            activation(),
            nn.ConvTranspose3d(in_channel, channel, 1),
            activation(),
            nn.ConvTranspose3d(channel, in_channel, 3, padding=1),
        )
        self.conv.apply(random_init)

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class ConvResnet3D(nn.Module):
    def __init__(self,
                 maxpool,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 strides,
                 dilatations,
                 padding,
                 batchnorm,
                 n_classes,
                 is_bayesian,
                 max_fvc,
                 random_node='output',
                 activation=torch.nn.ReLU,
                 n_res=3,
                 gated=True,
                 has_dense=True,
                 resblocks=False,
                 ):
        super().__init__()
        self.max_fvc = max_fvc
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.maxpool = nn.MaxPool3d(maxpool, return_indices=False)

        self.is_bayesian = is_bayesian
        if is_bayesian:
            if random_node == "output":
                self.GaussianSample = GaussianSample(1, 1)
            elif (random_node == "last"):
                self.GaussianSample = GaussianSample(1233, 1233)
        self.device = device
        self.conv_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.resconv = nn.ModuleList()
        self.activation = activation()

        self.n_res = n_res

        self.resblocks = resblocks
        self.has_dense = has_dense
        self.batchnorm = batchnorm
        self.a_dim = None
        for i, (ins, outs, ksize, stride, dilats, pad) in enumerate(zip(in_channels, out_channels,
                                                                        kernel_sizes, strides,
                                                                        dilatations, padding)):
            if not gated:
                self.conv_layers += [
                    torch.nn.Conv3d(in_channels=ins,
                                    out_channels=outs,
                                    kernel_size=ksize,
                                    stride=stride,
                                    padding=pad,
                                    dilation=dilats,
                                    )
                ]
            else:
                self.conv_layers += [
                    GatedConv3d(input_channels=ins,
                                output_channels=outs,
                                kernel_size=ksize,
                                stride=stride,
                                padding=pad,
                                dilation=dilats,
                                activation=nn.Tanh()
                                )]
            if resblocks and i != 0:
                for _ in range(n_res):
                    self.resconv += [ResBlock3D(ins, outs, activation, device)]
            self.bns += [nn.BatchNorm3d(num_features=outs)]
        self.dropout3d = nn.Dropout3d(0.5)
        self.dense1 = torch.nn.Linear(in_features=out_channels[-1], out_features=128)
        self.dense1_bn = nn.BatchNorm1d(num_features=128)
        self.dense2 = torch.nn.Linear(in_features=128 + 5, out_features=n_classes)  # 5 parameters added here
        self.dense2_bn = nn.BatchNorm1d(num_features=n_classes)
        self.dropout = nn.Dropout(0.5)
        self.log_softmax = torch.nn.functional.log_softmax

    def random_init(self, init_method=nn.init.xavier_uniform_):
        print("Random init")
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init_method(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, patient_info):
        j = 0
        for i in range(len(self.conv_layers)):
            if self.resblocks and i != 0:
                for _ in range(self.n_res):
                    x = self.resconv[j](x)
                    if self.batchnorm:
                        if x.shape[0] != 1:
                            x = self.bns[i - 1](x)
                    x = self.dropout3d(x)
                    j += 1
            x = self.conv_layers[i](x)
            if self.batchnorm:
                if x.shape[0] != 1:
                    x = self.bns[i](x)
            x = self.dropout3d(x)
            x = self.activation(x)
            x = self.maxpool(x)

        z = x.squeeze()
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        z = self.dense1(z)
        if self.batchnorm:
            if z.shape[0] != 1:
                z = self.dense1_bn(z)
        z = self.activation(z)
        z = self.dropout(z)
        z = torch.cat([z, patient_info], dim=1)
        z = self.dense2(z)
        z = torch.sigmoid_(z)
        if self.is_bayesian:
            z, mu, log_var = self.GaussianSample.float()(z)

        # if self.batchnorm:
        #     if z.shape[0] != 1:
        #         z = self.dense2_bn(z)
        # z = self.dropout(z)
        return z, mu, log_var

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

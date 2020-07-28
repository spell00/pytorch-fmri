import random
import argparse
import torch.nn as nn
import torch
from ..utils.masked_layer import GatedConv3d


def random_init(m, init_func=torch.nn.init.kaiming_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()



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
        out = self.conv.to(self.device)(input)
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
        out = self.conv.to(self.device)(input)
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
                 activation=torch.nn.ReLU,
                 n_res=3,
                 gated=True,
                 has_dense=True,
                 resblocks=False,
                 ):
        super().__init__()
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.maxpool = nn.MaxPool3d(maxpool, return_indices=False)

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
            self.dropout += [nn.Dropout3d(0.5)]
        self.dense1 = torch.nn.Linear(in_features=out_channels[-1], out_features=n_classes)
        self.dense1_bn = nn.BatchNorm1d(num_features=n_classes)
        self.dense1_dropout = nn.Dropout(0.5)

    def random_init(self, init_method=nn.init.xavier_normal_):
        print("Random init")
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init_method(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        j = 0
        for i in range(len(self.conv_layers)):
            if self.resblocks and i != 0:
                for _ in range(self.n_res):
                    if self.batchnorm:
                        if x.shape[0] != 1:
                            x = self.bns[i - 1].to(self.device)(x)
                    x = self.resconv[j](x)
                    j += 1
            x = self.conv_layers[i](x)
            if self.batchnorm:
                if x.shape[0] != 1:
                    x = self.bns[i].to(self.device)(x)
            x = self.activation(x)
            x = self.dropout[i](x)
            x = self.maxpool(x)

        z = x.squeeze()
        if self.has_dense:
            z = self.dense1(z)
            z = self.activation(z)
            if self.batchnorm:
                if z.shape[0] != 1:
                    z = self.dense1_bn(z)
            z = self.dense1_dropout(z)
        z = torch.softmax(z, 1)
        return z

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

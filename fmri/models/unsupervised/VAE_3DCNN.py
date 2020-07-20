import torch
from ..utils.stochastic import GaussianSample
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from ..utils.distributions import log_gaussian, log_standard_gaussian
from ..utils.flow import NormalizingFlows
from ..utils.masked_layer import GatedConv3d, GatedConvTranspose3d

in_channels = None
out_channels = None
kernel_sizes = None
strides = None


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparametrize(mu, log_var), mu, log_var


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class ResBlockDeconv(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(in_channel, channel, 1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(channel, in_channel, 3, padding=1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Autoencoder3DCNN(torch.nn.Module):
    def __init__(self,
                 z_dim,
                 maxpool,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 kernel_sizes_deconv,
                 strides,
                 strides_deconv,
                 dilatations,
                 dilatations_deconv,
                 padding,
                 padding_deconv,
                 batchnorm,
                 flow_type="nf",
                 n_flows=2,
                 n_res=3,
                 gated=True,
                 has_dense=True,
                 resblocks=False,
                 ):
        super(Autoencoder3DCNN, self).__init__()
        self.conv_layers = []
        self.deconv_layers = []
        self.bns = []
        self.resconv = []
        self.resconv1 = []
        self.resconv2 = []
        self.resblocks = resblocks
        self.resdeconv = []
        self.resdeconv1 = []
        self.resdeconv2 = []
        self.bns_deconv = []
        self.indices = [torch.Tensor() for _ in range(len(in_channels))]
        self.GaussianSample = GaussianSample(z_dim, z_dim)
        self.relu = torch.nn.LeakyReLU()
        self.has_dense = has_dense
        self.batchnorm = batchnorm
        self.n_res = n_res
        for ins, outs, ksize, stride, dilats, pad in zip(in_channels, out_channels,
                                                         kernel_sizes, strides,
                                                         dilatations, padding):
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
            if resblocks:
                for _ in range(n_res):
                    self.resconv += [ResBlock(ins, outs).cuda()]
            self.bns += [nn.BatchNorm3d(num_features=outs).cuda()]

        for ins, outs, ksize, stride, dilats, pad in zip(reversed(out_channels),
                                                         reversed(in_channels),
                                                         kernel_sizes_deconv,
                                                         strides_deconv,
                                                         dilatations_deconv,
                                                         padding_deconv):
            if not gated:
                self.deconv_layers += [torch.nn.ConvTranspose3d(in_channels=ins, out_channels=outs,
                                                                kernel_size=ksize, padding=pad, stride=stride,
                                                                dilation=dilats)]
            else:
                self.deconv_layers += [GatedConvTranspose3d(input_channels=ins, output_channels=outs,
                                                            kernel_size=ksize,
                                                            stride=stride, padding=pad, dilation=dilats,
                                                            activation=nn.Tanh()
                                                            )]
            if resblocks:
                for _ in range(n_res):
                    self.resdeconv += [ResBlockDeconv(ins, outs).cuda()]

            self.bns_deconv += [nn.BatchNorm3d(num_features=outs).cuda()]

        self.dense1 = torch.nn.Linear(in_features=out_channels[-1], out_features=z_dim)
        self.dense2 = torch.nn.Linear(in_features=z_dim, out_features=out_channels[-1])
        self.dense1_bn = nn.BatchNorm1d(num_features=z_dim)
        self.dense2_bn = nn.BatchNorm1d(num_features=out_channels[-1])
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool3d(maxpool, return_indices=True)
        self.maxunpool = nn.MaxUnpool3d(maxpool)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.deconv_layers = nn.ModuleList(self.deconv_layers)
        self.flow_type = flow_type
        self.n_flows = n_flows
        if self.flow_type == "nf":
            self.flow = NormalizingFlows(in_features=[z_dim], n_flows=n_flows)

    def random_init(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, mu, log_var):
        if len(z.shape) == 1:
            z = z.view(1, -1)
        if self.flow_type == "nf" and self.n_flows > 0:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)
        pz = log_standard_gaussian(z)

        kl = qz - pz

        return kl

    def encoder(self, x):
        j = 0
        for i in range(len(self.conv_layers)):
            if self.resblocks:
                for _ in range(self.n_res):
                    x = self.resconv[j](x)
                    j += 1
            x = self.conv_layers[i](x)
            if self.batchnorm:
                if x.shape[0] != 1:
                    x = self.bns[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            x, self.indices[i] = self.maxpool(x)

        z = x.squeeze()
        if self.has_dense:
            z = self.dense1(z)
            if self.batchnorm:
                if z.shape[0] != 1:
                    z = self.dense1_bn(z)
            z = self.relu(z)
            z = self.dropout(z)
        return z

    def decoder(self, z):
        if self.has_dense:
            z = self.dense2(z)
            z = self.relu(z)
            if self.batchnorm:
                if z.shape[0] != 1:
                    z = self.dense2_bn(z)
            z = self.dropout(z)

        j = 0
        x = z.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        for i in range(len(self.deconv_layers)):
            if self.resblocks:
                for _ in range(self.n_res):
                    x = self.resdeconv[j](x)
                    j += 1
            ind = self.indices[len(self.indices) - 1 - i]
            x = self.maxunpool(x[:, :, :ind.shape[2], :ind.shape[3], :ind.shape[4]], ind)
            x = self.deconv_layers[i](x)
            if self.batchnorm:
                if x.shape[0] != 1:
                   x = self.bns_deconv[i](x)
            if i < len(self.deconv_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)

        if (len(x.shape) == 3):
            x.unsqueeze(0)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):

        x = self.encoder(x)
        z, mu, log_var = self.GaussianSample(x)

        # Kullback-Leibler Divergence
        kl = self._kld(z, mu, log_var)

        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        rec = self.decoder(z)
        return rec, kl

    def sample(self, z, y=None):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

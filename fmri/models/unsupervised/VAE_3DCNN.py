import torch
from ..utils.stochastic import GaussianSample
import torch.nn as nn
import torch.nn.functional as F
from ..utils.distributions import log_gaussian, log_standard_gaussian
from ..utils.flow import NormalizingFlows, IAF, HouseholderFlow, ccLinIAF, SylvesterFlows, TriangularSylvester
from ..utils.masked_layer import GatedConv3d, GatedConvTranspose3d

in_channels = None
out_channels = None
kernel_sizes = None
strides = None


def swish(x):
    return x * x.sigmoid()


def mish(x):
    return x * F.softplus(x).tanh()


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return swish(x)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn(mu.size(), requires_grad=False)
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

    def __init__(self, in_features, out_features, device):
        super(GaussianSample, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features).to(device)
        self.log_var = nn.Linear(in_features, out_features).to(device)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparameterize(mu, log_var), mu, log_var


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, activation):
        super().__init__()

        self.conv = nn.Sequential(
            activation(),
            nn.Conv3d(in_channel, channel, 3, padding=1),
            activation(),
            nn.Conv3d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class ResBlockDeconv(nn.Module):
    def __init__(self, in_channel, channel, activation):
        super().__init__()

        self.conv = nn.Sequential(
            activation(),
            nn.ConvTranspose3d(in_channel, channel, 1),
            activation(),
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
                 activation,
                 flow_type="nf",
                 n_flows=2,
                 n_res=3,
                 gated=True,
                 has_dense=True,
                 resblocks=False,
                 device='cuda'
                 ):
        super(Autoencoder3DCNN, self).__init__()
        resconv = []
        resdeconv = []
        self.indices = [torch.Tensor() for _ in range(len(in_channels))]
        self.GaussianSample = GaussianSample(z_dim, z_dim, device)
        self.activation = activation()
        # self.swish = Swish()

        self.n_res = n_res

        self.resblocks = resblocks
        self.has_dense = has_dense
        self.batchnorm = batchnorm
        self.a_dim = None
        for i, (ins, outs, ksize, stride, dilats, pad) in enumerate(zip(in_channels,
                                                                        out_channels,
                                                                        kernel_sizes,
                                                                        strides,
                                                                        dilatations,
                                                                        padding)):
            layers_list = []
            if not gated:
                layers_list += [
                    nn.Conv3d(in_channels=ins,
                                    out_channels=outs,
                                    kernel_size=ksize,
                                    stride=stride,
                                    padding=pad,
                                    dilation=dilats,
                                    ),
                    nn.BatchNorm3d(num_features=outs),
                    activation(),
                    nn.Dropout3d(0.5),
                ]
            else:
                layers_list += [
                    GatedConv3d(input_channels=ins,
                                output_channels=outs,
                                kernel_size=ksize,
                                stride=stride,
                                padding=pad,
                                dilation=dilats,
                                activation=nn.Tanh()
                                ),
                    nn.BatchNorm3d(num_features=outs),
                    activation(),
                    nn.Dropout3d(0.5),
                ]
            if resblocks and i < len(in_channels) - 1:
                for _ in range(n_res):
                    layers_list += [
                        ResBlock(outs, outs, activation),
                        nn.BatchNorm3d(num_features=outs),
                        activation(),
                        nn.Dropout3d(0.5)
                    ]

            resconv += [nn.Sequential(*layers_list)]

        for i, (ins, outs, ksize, stride, dilats, pad) in enumerate(zip(reversed(out_channels),
                                                                        reversed(in_channels),
                                                                        kernel_sizes_deconv,
                                                                        strides_deconv,
                                                                        dilatations_deconv,
                                                                        padding_deconv)):
            layers_list = []
            if not gated:
                layers_list += [nn.ConvTranspose3d(in_channels=ins,
                                                         out_channels=outs,
                                                         kernel_size=ksize,
                                                         padding=pad,
                                                         stride=stride,
                                                         dilation=dilats),
                                nn.BatchNorm3d(num_features=outs),
                                activation(),
                                nn.Dropout3d(0.5),
                                ]
            else:
                layers_list += [GatedConvTranspose3d(input_channels=ins,
                                                     output_channels=outs,
                                                     kernel_size=ksize,
                                                     stride=stride,
                                                     padding=pad,
                                                     dilation=dilats,
                                                     activation=nn.Tanh()
                                                     ),
                                nn.BatchNorm3d(num_features=outs),
                                activation(),
                                nn.Dropout3d(0.5),
                                ]
            if resblocks and i < len(in_channels) - 1:
                for _ in range(n_res):
                    layers_list += [
                        ResBlockDeconv(outs, outs, activation),
                        nn.BatchNorm3d(num_features=outs),
                        activation(),
                        nn.Dropout3d(0.5)
                    ]
            resdeconv += [nn.Sequential(*layers_list)]

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=out_channels[-1], out_features=z_dim),
            nn.BatchNorm1d(num_features=z_dim),
            nn.Dropout(0.5)
        ).to(device)
        self.dense2 = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=out_channels[-1]),
            nn.BatchNorm1d(num_features=out_channels[-1]),
            nn.Dropout(0.5)
        ).to(device)
        self.maxpool = nn.MaxPool3d(maxpool, return_indices=True).to(device)
        self.maxunpool = nn.MaxUnpool3d(maxpool).to(device)
        self.resconv = nn.ModuleList(resconv).to(device)
        self.resdeconv = nn.ModuleList(resdeconv).to(device)
        self.flow_type = flow_type
        self.n_flows = n_flows
        if self.flow_type == "nf":
            self.flow = NormalizingFlows(in_features=[z_dim], n_flows=n_flows, device=device)
        if self.flow_type == "hf":
            self.flow = HouseholderFlow(in_features=[z_dim], auxiliary=False, n_flows=n_flows, h_last_dim=z_dim, device=device)
        if self.flow_type == "iaf":
            self.flow = IAF(z_dim, n_flows=n_flows, num_hidden=n_flows, h_size=z_dim, forget_bias=1., conv3d=False, device=device)
        if self.flow_type == "ccliniaf":
            self.flow = ccLinIAF(in_features=[z_dim], auxiliary=False, n_flows=n_flows, h_last_dim=z_dim, device=device)
        if self.flow_type == "o-sylvester":
            self.flow = SylvesterFlows(in_features=[z_dim], flow_flavour='o-sylvester', n_flows=1, h_last_dim=None, device=device)

    def random_init(self, func=nn.init.xavier_uniform_):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, h_last=None, p_param=None):
        if len(z.shape) == 1:
            z = z.view(1, -1)

        if (self.flow_type == "nf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif (self.flow_type == "iaf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z, h_last)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif (self.flow_type in ['hf', 'ccliniaf']) and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z = self.flow(z, h_last)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        elif self.flow_type in ["o-sylvester", "h-sylvester", "t-sylvester"] and self.n_flows > 0:
            mu, log_var, r1, r2, q_ortho, b = q_param
            f_z = self.flow(z, r1, r2, q_ortho, b)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        else:
            (mu, log_var) = q_param
            qz = log_gaussian(z, mu, log_var)
        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz

        return kl

    def encoder(self, x):
        for i, resconv in enumerate(self.resconv):
            x = resconv(x)
            x, self.indices[i] = self.maxpool(x)
        z = x.squeeze()
        if self.has_dense:
            z = self.dense1(z)
        return z

    def decoder(self, z):
        if self.has_dense:
            z = self.dense2(z)

        x = z.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        for i, resdeconv in enumerate(self.resdeconv):
            x = self.maxunpool(x, indices=self.indices[-i-1])
            x = resdeconv(x)

        if (len(x.shape) == 3):
            x.unsqueeze(0)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):

        x = self.encoder(x)
        z, mu, log_var = self.GaussianSample(x)

        # Kullback-Leibler Divergence
        kl = self._kld(z, (mu, log_var), x)

        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        rec = self.decoder(z)
        return rec, kl

    def sample(self, z):
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

import torch
from ..utils.stochastic import GaussianSample
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from ..utils.distributions import log_gaussian, log_standard_gaussian
from ..utils.flow import NormalizingFlows, IAF, HouseholderFlow, ccLinIAF, SylvesterFlows
from ..utils.masked_layer import GatedConv3d, GatedConvTranspose3d
from fmri.utils.quantizer import Quantize

in_channels = None
out_channels = None
kernel_sizes = None
strides = None


def random_init(m, init_func=torch.nn.init.orthogonal_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


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

        return self.reparameterize(mu, log_var), mu, log_var


class ResBlock(nn.Module):
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


class ResBlockDeconv(nn.Module):
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


class Autoencoder3DCNN(torch.nn.Module):
    def __init__(self,
                 z_dim,
                 maxpool,
                 # maxpool2,
                 in_channels,
                 # in_channels2,
                 out_channels,
                 # out_channels2,
                 kernel_sizes,
                 kernel_sizes_deconv,
                 strides,
                 strides_deconv,
                 dilatations,
                 dilatations_deconv,
                 padding,
                 # paddings2,
                 padding_deconv,
                 # paddings_deconv2,
                 batchnorm,
                 activation=torch.nn.ReLU,
                 flow_type="nf",
                 n_flows=2,
                 n_res=3,
                 n_embed=2000,
                 dropout_val=0.5,
                 gated=True,
                 has_dense=True,
                 resblocks=False,
                 ):
        super(Autoencoder3DCNN, self).__init__()

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.n_embed = n_embed
        self.out_channels = out_channels
        # self.out_channels2 = out_channels2
        self.device = device
        self.conv_layers = []
        self.deconv_layers = []
        # self.deconv_layers2 = []
        # self.conv_layers2 = []
        # self.bns2 = []
        # self.bns_deconv2 = []
        # self.resconv2 = []
        # self.resdeconv2 = []
        # self.indices2 = [torch.Tensor() for _ in range(len(in_channels2))]
        self.bns = []
        self.resconv = []
        self.resdeconv = []
        self.bns_deconv = []
        self.activations = []
        self.activation = activation()
        self.activation_deconv = activation()
        self.activations_deconv = []
        self.indices = [torch.Tensor() for _ in range(len(in_channels))]
        self.GaussianSample = GaussianSample(z_dim, z_dim)

        self.n_res = n_res

        self.resblocks = resblocks
        self.has_dense = has_dense
        self.batchnorm = batchnorm
        self.a_dim = None
        for i, (ins,
                # in2,
                outs,
                # out2,
                ksize,
                stride,
                dilats,
                pad,
                # pad2
                ) in enumerate(
            zip(in_channels,
                # in_channels2 + [None],
                out_channels,
                # out_channels2 + [None],
                kernel_sizes,
                strides,
                dilatations,
                padding,
                # paddings2 + [None]
                )):
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
                # if i < len(in_channels) - 1:
                #     self.conv_layers2 += [
                #         torch.nn.Conv3d(in_channels=in2,
                #                         out_channels=out2,
                #                         kernel_size=ksize,
                #                         stride=stride,
                #                         padding=pad2,
                #                         dilation=dilats,
                #                         )
                #     ]
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
                # if i < len(in_channels) - 1:
                #     self.conv_layers2 += [
                #         GatedConv3d(input_channels=in2,
                #                     output_channels=out2,
                #                     kernel_size=ksize,
                #                     stride=stride,
                #                     padding=pad2,
                #                     dilation=dilats,
                #                     activation=nn.Tanh()
                #                     )]
            if resblocks and i != 0:
                for _ in range(n_res):
                    self.resconv += [ResBlock(ins, outs, activation)]
                    # if i < len(in_channels) - 1:
                    #     self.resconv2 += [ResBlock(in2, out2, activation)]
            self.bns += [nn.BatchNorm3d(num_features=outs)]
            # if i < len(in_channels) - 1:
            #     self.bns2 += [nn.BatchNorm3d(num_features=out2)]
            self.activations += [activation()]
        for i, (ins, outs, ksize, stride, dilats, pad) in enumerate(zip(reversed(out_channels),
                                                                                         reversed(in_channels),
                                                                                         kernel_sizes_deconv,
                                                                                         strides_deconv,
                                                                                         dilatations_deconv,
                                                                                         padding_deconv)):
            if not gated:
                self.deconv_layers += [torch.nn.ConvTranspose3d(in_channels=ins, out_channels=outs,
                                                                kernel_size=ksize, padding=pad, stride=stride,
                                                                dilation=dilats)]
            #  if i < len(in_channels) - 1:
            #      self.deconv_layers2 += [torch.nn.ConvTranspose3d(in_channels=in2, out_channels=out2,
            #                                                   kernel_size=ksize, padding=pad2, stride=stride,
            #                                                  dilation=dilats)]
            else:
                self.deconv_layers += [GatedConvTranspose3d(input_channels=ins, output_channels=outs,
                                                            kernel_size=ksize,
                                                            stride=stride, padding=pad, dilation=dilats,
                                                            activation=nn.Tanh()
                                                            )]
                # if i < len(in_channels) - 1:
                #     self.deconv_layers2 += [GatedConvTranspose3d(input_channels=in2, output_channels=out2,
                #                                             kernel_size=ksize,
                #                                             stride=stride, padding=pad2, dilation=dilats,
                #                                              activation=nn.Tanh()
                #                                              )]
            if resblocks and i != 0:
                for _ in range(n_res):
                    self.resdeconv += [ResBlockDeconv(ins, outs, activation)]
                    # if i < len(in_channels) - 1:
                    #     self.resdeconv2 += [ResBlockDeconv(in2, out2, activation)]

            self.bns_deconv += [nn.BatchNorm3d(num_features=outs)]
            # if i < len(in_channels) - 1:
            #     self.bns_deconv2 += [nn.BatchNorm3d(num_features=out2)]
            self.activations_deconv += [activation()]

        self.dense1 = torch.nn.Linear(in_features=out_channels[-1], out_features=z_dim)
        self.dense2 = torch.nn.Linear(in_features=z_dim, out_features=out_channels[-1])
        self.dense1_bn = nn.BatchNorm1d(num_features=z_dim)
        self.dense2_bn = nn.BatchNorm1d(num_features=out_channels[-1])
        self.dropout3d = nn.Dropout3d(dropout_val)
        self.dropout = nn.Dropout(dropout_val)
        self.maxpool = nn.MaxPool3d(maxpool, return_indices=True)
        self.maxunpool = nn.MaxUnpool3d(maxpool)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.deconv_layers = nn.ModuleList(self.deconv_layers)
        self.bns = nn.ModuleList(self.bns)
        self.bns_deconv = nn.ModuleList(self.bns_deconv)
        self.resconv = nn.ModuleList(self.resconv)
        self.resdeconv = nn.ModuleList(self.resdeconv)
        # self.maxpool2 = nn.MaxPool3d(maxpool2, return_indices=True)
        # self.maxunpool2 = nn.MaxUnpool3d(3)
        # self.conv_layers2 = nn.ModuleList(self.conv_layers2)
        # self.deconv_layers2 = nn.ModuleList(self.deconv_layers2)
        # self.resconv2 = nn.ModuleList(self.resconv2)
        # self.bns_deconv2 = nn.ModuleList(self.bns_deconv2)
        # self.bns2 = nn.ModuleList(self.bns2)
        # self.resdeconv2 = nn.ModuleList(self.resdeconv2)
        self.flow_type = flow_type
        self.n_flows = n_flows
        if self.flow_type == "nf":
            self.flow = NormalizingFlows(in_features=[z_dim], n_flows=n_flows)
        if self.flow_type == "hf":
            self.flow = HouseholderFlow(in_features=[z_dim], auxiliary=False, n_flows=n_flows, h_last_dim=z_dim)
        if self.flow_type == "iaf":
            self.flow = IAF(z_dim, n_flows=n_flows, num_hidden=n_flows, h_size=z_dim, forget_bias=1., conv3d=False)
        if self.flow_type == "ccliniaf":
            self.flow = ccLinIAF(in_features=[z_dim], auxiliary=False, n_flows=n_flows, h_last_dim=z_dim)
        if self.flow_type == "o-sylvester":
            self.flow = SylvesterFlows(in_features=[z_dim], flow_flavour='o-sylvester', n_flows=1, h_last_dim=None)
        if self.flow_type == "quantizer":
            self.flow = Quantize(z_dim, self.n_embed)

    def random_init(self, init_func=torch.nn.init.orthogonal_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init_func(m.weight.data)
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
            x = self.activations[i](x)
            x, self.indices[i] = self.maxpool(x)

        z = x.squeeze()
        if self.has_dense:
            z = self.dense1(z)
            if self.batchnorm:
                if z.shape[0] != 1:
                    z = self.dense1_bn(z)
            z = self.dropout(z)
            z = self.activation(z)
        return z

    def decoder(self, z):
        if self.has_dense:
            z = self.dense2(z)
            if self.batchnorm:
                if z.shape[0] != 1:
                    z = self.dense2_bn(z)
            z = self.dropout(z)
            z = self.activation_deconv(z)

        j = 0
        x = z.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        for i in range(len(self.deconv_layers)):
            if self.resblocks and i != 0:
                for _ in range(self.n_res):
                    x = self.resdeconv[j](x)
                    if self.batchnorm:
                        if x.shape[0] != 1:
                            x = self.bns_deconv[i - 1](x)
                    x = self.dropout3d(x)
                    j += 1
            ind = self.indices[len(self.indices) - 1 - i]
            x = self.maxunpool(x[:, :, :ind.shape[2], :ind.shape[3], :ind.shape[4]], ind)
            x = self.deconv_layers[i](x)
            if i < len(self.deconv_layers) - 1:
                if self.batchnorm:
                    if x.shape[0] != 1:
                        x = self.bns_deconv[i](x)
                x = self.dropout3d(x)
                x = self.activations_deconv[i](x)

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

# https://github.com/ex4sperans/variational-inference-with-normalizing-flows/blob/master/flow.py
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from fmri.utils.utils import safe_log
import math

import torch as t
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter


class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()

        self.transforms = nn.Sequential(*(
            PlanarFlow(dim) for _ in range(flow_length)
        ))

        self.log_jacobians = nn.Sequential(*(
            PlanarFlowLogDetJacobian(t) for t in self.transforms
        ))

    def forward(self, z):

        log_jacobians = []

        for transform, log_jacobian in zip(self.transforms, self.log_jacobians):
            log_jacobians.append(log_jacobian(z))
            z = transform(z)

        zk = z

        return zk, log_jacobians


class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.tanh(activation)


class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return safe_log(det_grad.abs())


class linIAF(nn.Module):
    # https: // github.com / jmtomczak / vae_vpflows / blob / master / models / VAE_ccLinIAF.py  # L215
    def __init__(self, z1_size=2):
        super(linIAF, self).__init__()

        self.z1_size = z1_size
        self.cuda = torch.cuda.is_available()


    def forward(self, L, z):
        '''
        :param L: batch_size (B) x latent_size^2 (L^2)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = L*z
        '''
        # L->tril(L)
        L_matrix = L.view( -1, self.z1_size, self.z1_size ) # resize to get B x L x L
        LTmask = torch.tril( torch.ones(self.z1_size, self.z1_size), diagonal=-1 ) # lower-triangular mask matrix (1s in lower triangular part)
        I = Variable( torch.eye(self.z1_size, self.z1_size).expand(L_matrix.size(0), self.z1_size, self.z1_size) )
        if self.cuda:
            LTmask = LTmask.cuda()
            I = I.cuda()
        LTmask = Variable(LTmask)
        LTmask = LTmask.unsqueeze(0).expand( L_matrix.size(0), self.z1_size, self.z1_size ) # 1 x L x L -> B x L x L
        LT = torch.mul( L_matrix, LTmask ) + I # here we get a batch of lower-triangular matrices with ones on diagonal

        # z_new = L * z
        z_new = torch.bmm( LT , z.unsqueeze(2) ).squeeze(2) # B x L x L * B x L x 1 -> B x L

        return z_new


class HF(nn.Module):
        def __init__(self):
            super(HF, self).__init__()

        def forward(self, v, z):
            '''
            :param v: batch_size (B) x latent_size (L)
            :param z: batch_size (B) x latent_size (L)
            :return: z_new = z - 2* v v_T / norm(v,2) * z
            '''
            # v * v_T
            vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L
            # v * v_T * z
            vvTz = torch.bmm(vvT, z.unsqueeze(2)).squeeze(2)  # A * z : batchdot( B x L x L * B x L x 1 ).squeeze(2) = (B x L x 1).squeeze(2) = B x L
            # calculate norm ||v||^2
            norm_sq = torch.sum(v * v, 1, keepdim=True)  # calculate norm-2 for each row : B x 1
            norm_sq = norm_sq.expand(norm_sq.size(0), v.size(1))  # expand sizes : B x L
            # calculate new z
            z_new = z - 2 * vvTz / norm_sq  # z - 2 * v * v_T  * z / norm2(v)
            return z_new


class combination_L(nn.Module):
    # https: // github.com / jmtomczak / vae_vpflows / blob / master / models / VAE_ccLinIAF.py  # L215
    def __init__(self,z1_size,number_combination):
        super(combination_L, self).__init__()
        self.z1_size = z1_size
        self.number_combination = number_combination

    def forward(self, L, y):
        '''
        :param L: batch_size (B) x latent_size^2 * number_combination (L^2 * C)
        :param y: batch_size (B) x number_combination (C)
        :return: L_combination = y * L
        '''
        # calculate combination of Ls
        L_tensor = L.view( -1, self.z1_size**2, self.number_combination ) # resize to get B x L^2 x C
        y = y.unsqueeze(1).expand(y.size(0), self.z1_size**2, y.size(1)) # expand to get B x L^2 x C
        L_combination = torch.sum( L_tensor * y, 2 ).squeeze()
        return L_combination


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, h, g):
        return h * g


class AutoregressiveLinear(nn.Module):
    def __init__(self, in_size, out_size, bias=True, ):
        super(AutoregressiveLinear, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.weight = Parameter(t.Tensor(self.in_size, self.out_size))

        if bias:
            self.bias = Parameter(t.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = xavier_normal(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return t.addmm(self.bias, input, self.weight.tril(-1))

        output = input @ self.weight.tril(-1)
        if self.bias is not None:
            output += self.bias
        return output
import functools
from math import exp

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from torchvision import models
import torch.utils.model_zoo as model_zoo


################################## ResnetBlock ##################################
class ResnetBlockSpectralNorm(nn.Module):
    def __init__(self, dim, padding_type, activation=nn.LeakyReLU(0.2), use_dropout=False):
        super(ResnetBlockSpectralNorm, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=p)),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=p))]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
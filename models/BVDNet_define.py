import numpy as np
from .model_util import *
from models import model_util
from models.BVDNet import BVDNet


def define_G(N=2, n_blocks=1, gpu_id='-1') -> BVDNet:
    netG = BVDNet(N=N, n_blocks=n_blocks)
    netG = model_util.todevice(netG, gpu_id)
    netG.apply(model_util.init_weights)
    return netG


################################Discriminator################################
def define_D(input_nc=6, ndf=64, n_layers_D=1, use_sigmoid=False, num_D=3, gpu_id='-1'):
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid, num_D)
    netD = model_util.todevice(netD, gpu_id)
    netD.apply(model_util.init_weights)
    return netD


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, use_sigmoid)
            setattr(self, 'layer' + str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        return self.model(input)


class GANLoss(nn.Module):
    def __init__(self, mode='D'):
        super(GANLoss, self).__init__()
        if mode == 'D':
            self.lossf = model_util.HingeLossD()
        elif mode == 'G':
            self.lossf = model_util.HingeLossG()
        self.mode = mode

    def forward(self, dis_fake=None, dis_real=None):
        if isinstance(dis_fake, list):
            if self.mode == 'D':
                loss = 0
                for i in range(len(dis_fake)):
                    loss += self.lossf(dis_fake[i][-1], dis_real[i][-1])
            elif self.mode == 'G':
                loss = 0
                weight = 2 ** len(dis_fake)
                for i in range(len(dis_fake)):
                    weight = weight / 2
                    loss += weight * self.lossf(dis_fake[i][-1])
            return loss
        else:
            if self.mode == 'D':
                return self.lossf(dis_fake[-1], dis_real[-1])
            elif self.mode == 'G':
                return self.lossf(dis_fake[-1])

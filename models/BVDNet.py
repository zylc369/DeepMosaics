import torch
import torch.nn as nn

from .ResnetBlockSpectralNorm import ResnetBlockSpectralNorm
import torch.nn.utils.spectral_norm as SpectralNorm

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super().__init__()

        self.convup = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.ReflectionPad2d(padding),
                # EqualConv2d(out_channel, out_channel, kernel_size, padding=padding),
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size)),
                nn.LeakyReLU(0.2),
                # Blur(out_channel),
            )

    def forward(self, input):
        outup = self.convup(input)
        return outup

class Encoder2d(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, activation = nn.LeakyReLU(0.2)):
        super(Encoder2d, self).__init__()        
   
        model = [nn.ReflectionPad2d(3), SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [  nn.ReflectionPad2d(1),
                        SpectralNorm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0)), 
                        activation]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class Encoder3d(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, activation = nn.LeakyReLU(0.2)):
        super(Encoder3d, self).__init__()        
               
        model = [SpectralNorm(nn.Conv3d(input_nc, ngf, kernel_size=3, padding=1)), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [  SpectralNorm(nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)),
                         activation]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class BVDNet(nn.Module):
    def __init__(self, N=2, n_downsampling=3, n_blocks=4, input_nc=3, output_nc=3,activation=nn.LeakyReLU(0.2)):
        super(BVDNet, self).__init__()
        ngf = 64
        padding_type = 'reflect'
        self.N = N

        ### encoder
        self.encoder3d = Encoder3d(input_nc,64,n_downsampling,activation)
        self.encoder2d = Encoder2d(input_nc,64,n_downsampling,activation)

        ### resnet blocks
        self.blocks = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            self.blocks += [ResnetBlockSpectralNorm(ngf * mult, padding_type=padding_type, activation=activation)]
        self.blocks = nn.Sequential(*self.blocks)

        ### decoder
        self.decoder = []        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            self.decoder += [UpBlock(ngf * mult, int(ngf * mult / 2))]
        self.decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]        
        self.decoder = nn.Sequential(*self.decoder)
        self.limiter = nn.Tanh()

    def forward(self, stream, previous):
        this_shortcut = stream[:,:,self.N]
        stream = self.encoder3d(stream)
        stream = stream.reshape(stream.size(0),stream.size(1),stream.size(3),stream.size(4))
        previous = self.encoder2d(previous)
        x = stream + previous
        x = self.blocks(x)
        x = self.decoder(x)
        x = x+this_shortcut
        x = self.limiter(x)
        return x

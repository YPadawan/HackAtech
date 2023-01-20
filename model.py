import torch
import helper
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


class ConvBNReluBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, resampling_filter=None, padding=1, upsample=False,
                 downsample=False):
        super().__init__()

        ### Convolution layer

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.upsample = upsample
        self.downsample = downsample

        self.resampling_filter = resampling_filter

    def forward(self, x):
        x = self.block(x)

        if self.downsample:
            x = F.dropout2d(x, p=0.05)
            x = F.max_pool2d(x, kernel_size=self.resampling_filter)

        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=self.resampling_filter)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cbe1 = ConvBNReluBlock(1, 32, kernel_size=(3, 2), resampling_filter=(5, 2), downsample=True)
        self.cbe2 = ConvBNReluBlock(32, 128, kernel_size=(3, 2), resampling_filter=(4, 2), downsample=True)
        self.cbe3 = ConvBNReluBlock(128, 256, kernel_size=(3, 2), resampling_filter=(2, 2), downsample=True)

    def forward(self, x):
        x = self.cbe1(x)
        x = self.cbe2(x)
        x = self.cbe3(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.convbnrelu1 = ConvBNReluBlock(in_channels, in_channels, kernel_size=3)

        self.convbnrelu2 = ConvBNReluBlock(in_channels, in_channels, kernel_size=3)

        self.convbnrelu3 = ConvBNReluBlock(in_channels, in_channels, kernel_size=3)

    def forward(self, x):
        residual = x

        x = self.convbnrelu1(x)
        x = self.convbnrelu2(x)
        x = self.convbnrelu3(x)

        return x + residual


class ResNet(nn.Module):
    def __init__(self, in_channels, num_blocks):
        super(ResNet, self).__init__()
        self.blocks = [ResidualBlock(in_channels=256) for i in range(num_blocks)]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.convbnrelu1 = ConvBNReluBlock(256, 128, kernel_size=(3, 2), resampling_filter=(5, 3), padding=1,
                                           upsample=True)
        self.convbnrelu2 = ConvBNReluBlock(128, 32, kernel_size=(3, 2), resampling_filter=(4, 1), padding=1,
                                           upsample=True)
        self.convbnrelu3 = ConvBNReluBlock(32, 1, kernel_size=(3, 2), resampling_filter=(2, 2), padding=1,
                                           upsample=True)

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.convbnrelu2(x)
        x = self.convbnrelu3(x)

        return x
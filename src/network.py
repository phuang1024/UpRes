import torch
from torch.nn import Module

from constants import *


class Conv(Module):
    """
    Convolution, batch norm, and relu.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvPooling(Module):
    """
    Convolution, batch norm, pooling, and relu.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.pool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class DenseBlock(Module):
    """
    Output channels = input channels
    """

    def __init__(self, channels, depth=2):
        super().__init__()
        self.depth = depth

        curr_ch = channels
        for i in range(depth):
            setattr(self, f"conv{i}", Conv(curr_ch, curr_ch))
            curr_ch += curr_ch
        self.conv_out = Conv(curr_ch, channels)

    def forward(self, x):
        for i in range(self.depth):
            conv = getattr(self, f"conv{i}")
            x = torch.cat([x, conv(x)], dim=1)
        x = self.conv_out(x)
        return x


class RRDB(Module):
    """
    Residual in residual dense block.
    Chain of dense blocks.

    Output channels = input channels * 2
    """

    def __init__(self, channels, depth=2):
        super().__init__()
        self.depth = depth

        for i in range(depth):
            setattr(self, f"dense_block{i}", DenseBlock(channels))
            beta = torch.nn.Parameter(torch.ones(1))
            beta.requires_grad = True
            setattr(self, f"beta{i}", beta)
        self.beta_out = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        original = x
        for i in range(self.depth):
            dense_block = getattr(self, f"dense_block{i}")
            beta = getattr(self, f"beta{i}")
            x = x + beta * dense_block(x)
        x = x * getattr(self, "beta_out")
        x = torch.cat([x, original], dim=1)
        return x


class UpresNet(Module):
    """
    Implementation of ESRGAN
    """

    def __init__(self):
        super().__init__()

        self.conv_in = torch.nn.Sequential(
            Conv(3, 4),
            Conv(4, 8),
            Conv(8, 16),
        )

        self.rrdb = torch.nn.Sequential(
            RRDB(16),
            RRDB(32),
            #RRDB(64),
        )

        # Use number of layers to control scale factor.
        self.upsamp = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 16, 4, 2, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
        )

        self.conv_out = torch.nn.Sequential(
            Conv(16, 8),
            Conv(8, 4),
            Conv(4, 3, 1),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.rrdb(x)
        x = self.upsamp(x)
        x = self.conv_out(x)
        x = self.sigmoid(x)
        return x


class Discriminator(Module):
    """
    Discriminator.
    Output 0 = fake, 1 = real
    """

    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Sequential(
            ConvPooling(3, 8, 5),
            ConvPooling(8, 16, 5),
            ConvPooling(16, 32, 5),
            ConvPooling(32, 64, 5),
            ConvPooling(64, 32, 5),
            ConvPooling(32, 16, 5),
            ConvPooling(16, 8, 5),
            ConvPooling(8, 4, 5),
        )
        self.head = Conv(4, 1, 4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        x = self.sigmoid(x)
        return x

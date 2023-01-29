import torch
from torch.nn import Module

from constants import *


class Conv(Module):
    """
    Convolution and some stuff.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class DenseBlock(Module):
    def __init__(self, channels, depth=3):
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
    """

    def __init__(self, channels, depth=3):
        super().__init__()
        self.depth = depth

        for i in range(depth):
            setattr(self, f"dense_block{i}", DenseBlock(channels))
            beta = torch.nn.Parameter(torch.ones(1))
            beta.requires_grad = True
            setattr(self, f"beta{i}", beta)
        setattr(self, "beta_out", torch.nn.Parameter(torch.ones(1)))

    def forward(self, x):
        for i in range(self.depth):
            dense_block = getattr(self, f"dense_block{i}")
            beta = getattr(self, f"beta{i}")
            x = beta * dense_block(x)
        x = x * getattr(self, "beta_out")
        return x


class UpresNet(Module):
    """
    Implementation of ESRGAN
    """

    def __init__(self, block_depth=3):
        super().__init__()

        self.block_depth = block_depth

        self.conv_in = torch.nn.Sequential(
            Conv(3, 4),
            Conv(4, 8),
            Conv(8, 16),
        )

        for i in range(block_depth):
            setattr(self, f"rrdb{i}", RRDB(16))

        # Use number of layers to control scale factor.
        self.upsamp = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, 4, 2, 1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(8, 4, 4, 2, 1),
            torch.nn.LeakyReLU(),
        )

        self.conv_out = torch.nn.Sequential(
            Conv(4, 3, 1),
        )

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.block_depth):
            rrdb = getattr(self, f"rrdb{i}")
            x = rrdb(x)
        x = self.upsamp(x)
        x = self.conv_out(x)
        return x

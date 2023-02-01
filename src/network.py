import torch
from torch.nn import Module, BatchNorm2d

from constants import *


class Conv(Module):
    """
    Convolution, batch norm, and relu.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        padding = padding if padding is not None else kernel_size//2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.bn = BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTranspose(Module):
    """
    ConvTranspose, batch norm, and relu.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        padding = padding if padding is not None else kernel_size//2
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.bn = BatchNorm2d(out_channels)
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
        self.bn = BatchNorm2d(out_channels)
        self.pool = torch.nn.AvgPool2d(2)
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
            beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            setattr(self, f"beta{i}", beta)
        self.beta_out = torch.nn.Parameter(torch.ones(1), requires_grad=True)

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
            Conv(16, 32),
        )

        self.rrdb = torch.nn.Sequential(
            RRDB(32),
            RRDB(64),
            RRDB(128),
        )

        # Use number of layers to control scale factor.
        self.upsamp = torch.nn.Sequential(
            ConvTranspose(256, 128, 4, 2, 1),
            ConvTranspose(128, 64, 4, 2, 1),
            torch.nn.BatchNorm2d(64),
        )

        self.conv_out = torch.nn.Sequential(
            Conv(64, 32),
            Conv(32, 16),
            Conv(16, 8),
            Conv(8, 4),
        )
        self.head = torch.nn.Conv2d(4, 3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.rrdb(x)
        x = self.upsamp(x)
        x = self.conv_out(x)
        x = self.head(x)
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
            Conv(3, 8, 4, 2, 1),
            #Conv(8, 16, 4, 2, 1),
            #Conv(16, 32, 4, 2, 1),
            #Conv(32, 64, 4, 2, 1),
            #Conv(64, 128, 4, 2, 1),
            #Conv(128, 256, 4, 2, 1),
        )
        self.head = torch.nn.Linear(8*128*128, 1)
        #self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        #x = self.sigmoid(x)
        return x

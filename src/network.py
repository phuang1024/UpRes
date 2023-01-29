import cv2
import torch

from constants import *


class Conv(torch.nn.Module):
    """
    Convolution and some stuff.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class UpresNet(torch.nn.Module):
    """
    RGB image input and output.
    - conv
    - upsample
    - conv
    - head
    """

    def __init__(self, kernel_size=3):
        super().__init__()

        self.scale_fac = SCALE_FAC
        self.kernel_size = kernel_size

        self.upsample = torch.nn.Upsample(scale_factor=SCALE_FAC, mode="bilinear", align_corners=False)
        self.convs = torch.nn.Sequential(
            Conv(3, 8, kernel_size),
            Conv(8, 16, kernel_size),
            Conv(16, 32, kernel_size),
        )
        self.conv_trans = torch.nn.ConvTranspose2d(32, 32, kernel_size, padding=kernel_size//2, stride=SCALE_FAC)
        self.convs2 = torch.nn.Sequential(
            Conv(32, 16, kernel_size),
            Conv(16, 8, kernel_size),
            Conv(8, 4, kernel_size),
        )
        self.head = Conv(4+3, 3, 1)

    def forward(self, x):
        upscale = self.upsample(x)
        x = self.convs(x)
        x = self.conv_trans(x, output_size=upscale.shape)
        x = self.convs2(x)
        x = torch.cat([upscale, x], dim=1)
        x = self.head(x)
        return x

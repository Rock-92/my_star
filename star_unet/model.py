from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedConv(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=int(dilation),
                dilation=int(dilation),
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedBottleneck(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(DilatedConv(channels, dilation) for dilation in dilations)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(dilations), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [block(x) for block in self.blocks]
        return self.fuse(torch.cat(features, dim=1)) + x


class UNet(nn.Module):
    """Shallow U-Net with 2x downsampling and a dilated bottleneck."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64),
    ) -> None:
        super().__init__()
        if len(features) != 2:
            raise ValueError("2x-dilated UNet expects exactly two feature sizes, for example (32, 64)")

        low_channels, high_channels = int(features[0]), int(features[1])
        self.enc1 = DoubleConv(in_channels, low_channels)
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(low_channels, high_channels),
        )
        self.bottleneck = DilatedBottleneck(high_channels)

        self.dec_half = DoubleConv(high_channels + high_channels, high_channels)
        self.up = nn.ConvTranspose2d(high_channels, low_channels, kernel_size=2, stride=2)
        self.dec_full = DoubleConv(low_channels + low_channels, low_channels)
        self.outc = nn.Conv2d(low_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_full = self.enc1(x)
        skip_half = self.down(skip_full)

        x = self.bottleneck(skip_half)
        x = self.dec_half(torch.cat([skip_half, x], dim=1))
        x = self.up(x)

        dy = skip_full.size(2) - x.size(2)
        dx = skip_full.size(3) - x.size(3)
        if dx != 0 or dy != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])

        x = self.dec_full(torch.cat([skip_full, x], dim=1))
        return self.outc(x)

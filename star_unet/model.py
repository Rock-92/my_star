from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions used by the vanilla U-Net blocks."""

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


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dx != 0 or dy != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Vanilla U-Net for single-channel star segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        if len(features) < 2:
            raise ValueError("features must contain at least two channel sizes")

        self.inc = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList(
            Down(features[i], features[i + 1]) for i in range(len(features) - 1)
        )
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        decoder_in = features[-1] * 2
        self.ups = nn.ModuleList()
        for skip_channels in reversed(features):
            self.ups.append(Up(decoder_in, skip_channels, skip_channels))
            decoder_in = skip_channels

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.outc(x)

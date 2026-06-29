from __future__ import annotations

import torch
import torch.nn as nn


class DeepSourceEnhancer(nn.Module):
    """Small fully-convolutional SNR enhancer from the DeepSource paper/code.

    The network keeps input/output resolution identical:
    Conv5x5 -> Conv5x5 -> Conv5x5 -> residual add -> BN -> Conv5x5
    -> Dropout -> Conv5x5(1).
    """

    def __init__(
        self,
        in_channels: int = 1,
        filters: int = 16,
        kernel_size: int = 5,
        dropout: float = 0.25,
        output_activation: str = "relu",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(filters)
        self.conv4 = nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout2d(float(dropout))
        self.out = nn.Conv2d(filters, 1, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.output_activation = str(output_activation).lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.relu(self.conv1(x))
        x = self.relu(self.conv2(skip))
        x = self.relu(self.conv3(x))
        x = self.bn(x + skip)
        x = self.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.out(x)
        if self.output_activation == "relu":
            return self.relu(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation in {"none", "linear"}:
            return x
        raise ValueError(f"unsupported output_activation: {self.output_activation}")


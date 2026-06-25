from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateScorer(nn.Module):
    """Legacy binary scorer kept for existing checkpoints."""

    def __init__(self, input_channels: int = 1, feature_dim: int = 0) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.features = nn.Sequential(
            nn.Conv2d(int(input_channels), 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        if self.feature_dim > 0:
            self.numeric = nn.Sequential(
                nn.Linear(self.feature_dim, 32),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(32),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )
            classifier_in = 96
        else:
            self.numeric = None
            classifier_in = 64
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, numeric_features: torch.Tensor | None = None) -> torch.Tensor:
        image_features = torch.flatten(self.features(x), 1)
        if self.numeric is not None:
            if numeric_features is None:
                raise ValueError("numeric_features is required when feature_dim > 0")
            image_features = torch.cat([image_features, self.numeric(numeric_features)], dim=1)
        return self.classifier(image_features).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class PatchEncoder(nn.Module):
    def __init__(self, input_channels: int, width: int = 24) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            ResidualBlock(width, width),
            ResidualBlock(width, width * 2, stride=2),
            ResidualBlock(width * 2, width * 2),
            ResidualBlock(width * 2, width * 4, stride=2),
            ResidualBlock(width * 4, width * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output_dim = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.network(x), 1)


class CenterAwareScorer(nn.Module):
    """Dual-scale scorer with class, quality, and sub-pixel offset heads."""

    def __init__(self, input_channels: int = 6, feature_dim: int = 16, width: int = 24) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.feature_dim = int(feature_dim)
        self.width = int(width)
        self.small_encoder = PatchEncoder(self.input_channels, self.width)
        self.large_encoder = PatchEncoder(self.input_channels, self.width)
        numeric_dim = 64 if self.feature_dim else 0
        self.numeric = (
            nn.Sequential(
                nn.Linear(self.feature_dim, 64),
                nn.LayerNorm(64),
                nn.SiLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, 64),
                nn.SiLU(inplace=True),
            )
            if self.feature_dim
            else None
        )
        fused_dim = self.small_encoder.output_dim + self.large_encoder.output_dim + numeric_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(192, 128),
            nn.SiLU(inplace=True),
        )
        self.class_head = nn.Linear(128, 3)
        self.quality_head = nn.Linear(128, 1)
        self.offset_head = nn.Linear(128, 2)

    def forward(
        self,
        patches_small: torch.Tensor,
        patches_large: torch.Tensor,
        numeric_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = [self.small_encoder(patches_small), self.large_encoder(patches_large)]
        if self.numeric is not None:
            if numeric_features is None:
                raise ValueError("numeric_features is required")
            features.append(self.numeric(numeric_features))
        fused = self.fusion(torch.cat(features, dim=1))
        return {
            "class_logits": self.class_head(fused),
            "quality_logit": self.quality_head(fused).squeeze(1),
            "offset_yx": self.offset_head(fused),
        }


def build_model_from_checkpoint(checkpoint: dict[str, object]) -> nn.Module:
    model_type = str(checkpoint.get("model_type", "legacy"))
    if model_type == "center_aware_v2":
        return CenterAwareScorer(
            input_channels=int(checkpoint.get("input_channels", 6)),
            feature_dim=int(checkpoint.get("feature_dim", 16)),
            width=int(checkpoint.get("model_width", 24)),
        )
    return CandidateScorer(
        input_channels=int(checkpoint.get("input_channels", 1)),
        feature_dim=int(checkpoint.get("feature_dim", 0)),
    )

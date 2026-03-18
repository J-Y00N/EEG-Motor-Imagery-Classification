"""EEGNet architecture used for deep learning baselines."""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    """Compact EEGNet for left-vs-right motor imagery classification."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        dropout_rate: float = 0.5,
        kernel_length: int = 64,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            features = self.block2(self.block1(dummy))
            flattened_dim = features.reshape(1, -1).shape[1]

        self.classifier = nn.Linear(flattened_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


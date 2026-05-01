
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: Sequence[int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        if len(features) != 4:
            raise ValueError(f"Expected 4 feature levels, got {len(features)}: {features}")

        f0, f1, f2, f3 = (int(f) for f in features)

        self.enc1 = _DoubleConv(in_channels, f0)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = _DoubleConv(f0, f1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = _DoubleConv(f1, f2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = _DoubleConv(f2, f3)

        self.up3 = nn.ConvTranspose3d(f3, f2, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(f2 + f2, f2)

        self.up2 = nn.ConvTranspose3d(f2, f1, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(f1 + f1, f1)

        self.up1 = nn.ConvTranspose3d(f1, f0, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(f0 + f0, f0)

        self.out_conv = nn.Conv3d(f0, out_channels, kernel_size=1)

    @staticmethod
    def _crop_like(skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Center-crop skip to match x spatial shape (D,H,W)."""
        if skip.shape[2:] == x.shape[2:]:
            return skip

        sd, sh, sw = skip.shape[2:]
        xd, xh, xw = x.shape[2:]

        dd = max(0, (sd - xd) // 2)
        dh = max(0, (sh - xh) // 2)
        dw = max(0, (sw - xw) // 2)

        return skip[:, :, dd : dd + xd, dh : dh + xh, dw : dw + xw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        xb = self.bottleneck(self.pool3(x3))

        u3 = self.up3(xb)
        s3 = self._crop_like(x3, u3)
        d3 = self.dec3(torch.cat([s3, u3], dim=1))

        u2 = self.up2(d3)
        s2 = self._crop_like(x2, u2)
        d2 = self.dec2(torch.cat([s2, u2], dim=1))

        u1 = self.up1(d2)
        s1 = self._crop_like(x1, u1)
        d1 = self.dec1(torch.cat([s1, u1], dim=1))

        # NOTE:
        # This returns raw logits.
        # Apply softmax during inference/visualization:
        # torch.softmax(output, dim=1)
        return self.out_conv(d1)

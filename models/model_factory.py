from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch.nn as nn

from models.swinunetr import SwinUNETRConfig, build_swinunetr
from models.unet3d import BaselineUNet3D, ResidualUNet3D


def build_model(
    model_name: str,
    *,
    in_channels: int,
    out_channels: int,
    patch_size: tuple[int, int, int],
    baseline_features: Sequence[int] = (16, 32, 64, 128),
    residual_features: Sequence[int] = (32, 64, 128, 256),
    swin_feature_size: int = 24,
    swin_use_checkpoint: bool = True,
) -> nn.Module:
    name = str(model_name).lower().strip()

    if name == "baseline_unet":
        return BaselineUNet3D(in_channels=int(in_channels), out_channels=int(out_channels), features=tuple(baseline_features))

    if name == "residual_unet":
        return ResidualUNet3D(in_channels=int(in_channels), out_channels=int(out_channels), features=tuple(residual_features))

    if name == "swinunetr":
        return build_swinunetr(
            SwinUNETRConfig(
                img_size=tuple(int(v) for v in patch_size),
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                feature_size=int(swin_feature_size),
                use_checkpoint=bool(swin_use_checkpoint),
            )
        )

    raise ValueError(f"Unknown model_name: {model_name}. Expected one of: baseline_unet, residual_unet, swinunetr")


def model_metadata(
    model_name: str,
    *,
    baseline_features: Sequence[int],
    residual_features: Sequence[int],
    swin_feature_size: int,
    swin_use_checkpoint: bool,
) -> Mapping[str, Any]:
    name = str(model_name).lower().strip()
    if name == "baseline_unet":
        return {"model_name": name, "baseline_features": [int(v) for v in baseline_features]}
    if name == "residual_unet":
        return {"model_name": name, "residual_features": [int(v) for v in residual_features]}
    if name == "swinunetr":
        return {"model_name": name, "swin_feature_size": int(swin_feature_size), "swin_use_checkpoint": bool(swin_use_checkpoint)}
    return {"model_name": name}


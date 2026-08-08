from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch.nn as nn

from models.swinunetr import (
    PretrainedLoadReport,
    SwinUNETRConfig,
    build_swinunetr,
    load_pretrained_swin_weights,
    set_swin_encoder_trainable,
)
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
    use_pretrained_swin: bool = False,
    swin_pretrained_path: str | None = None,
    swin_pretrained_url: str | None = None,
    freeze_swin_encoder: bool = False,
) -> tuple[nn.Module, Optional[PretrainedLoadReport]]:
    """
    Build a segmentation model.

    Returns:
        (model, pretrained_report) — report is non-None only for swinunetr when pretrained was attempted.
    """
    name = str(model_name).lower().strip()
    report: Optional[PretrainedLoadReport] = None

    if name == "baseline_unet":
        return (
            BaselineUNet3D(
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                features=tuple(baseline_features),
            ),
            None,
        )

    if name == "residual_unet":
        return (
            ResidualUNet3D(
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                features=tuple(residual_features),
            ),
            None,
        )

    if name == "swinunetr":
        model = build_swinunetr(
            SwinUNETRConfig(
                img_size=tuple(int(v) for v in patch_size),
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                feature_size=int(swin_feature_size),
                use_checkpoint=bool(swin_use_checkpoint),
            )
        )
        if use_pretrained_swin and swin_pretrained_path:
            report = load_pretrained_swin_weights(
                model,
                weights_path=str(swin_pretrained_path),
                download_url=swin_pretrained_url,
            )
        if freeze_swin_encoder:
            n = set_swin_encoder_trainable(model, trainable=False)
            print(f"[model_factory] Froze {n} Swin encoder parameters (decoder/head remain trainable)")
        return model, report

    raise ValueError(
        f"Unknown model_name: {model_name}. Expected one of: baseline_unet, residual_unet, swinunetr"
    )


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
        return {
            "model_name": name,
            "swin_feature_size": int(swin_feature_size),
            "swin_use_checkpoint": bool(swin_use_checkpoint),
            "deep_supervision": "NOT IMPLEMENTED / FUTURE EXPERIMENT",
        }
    return {"model_name": name}

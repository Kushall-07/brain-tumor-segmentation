from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch.nn as nn


@dataclass(frozen=True)
class SwinUNETRConfig:
    img_size: tuple[int, int, int] = (96, 96, 96)
    in_channels: int = 4
    out_channels: int = 3

    # Memory/perf tuning (6GB-friendly defaults).
    feature_size: int = 24
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    use_checkpoint: bool = True


def build_swinunetr(cfg: SwinUNETRConfig) -> nn.Module:
    """
    Build MONAI SwinUNETR for 3D segmentation.

    Notes:
    - `img_size` must match your training/inference ROI (here: 96^3).
    - `use_checkpoint=True` reduces VRAM at the cost of speed.
    """
    try:
        from monai.networks.nets import SwinUNETR  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MONAI SwinUNETR is not available. Ensure `monai` is installed with the correct version."
        ) from e

    return SwinUNETR(
        img_size=tuple(int(v) for v in cfg.img_size),
        in_channels=int(cfg.in_channels),
        out_channels=int(cfg.out_channels),
        feature_size=int(cfg.feature_size),
        drop_rate=float(cfg.drop_rate),
        attn_drop_rate=float(cfg.attn_drop_rate),
        dropout_path_rate=float(cfg.dropout_path_rate),
        use_checkpoint=bool(cfg.use_checkpoint),
    )


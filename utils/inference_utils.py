from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference


def _flip_spatial_dims(tensor: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    if not dims:
        return tensor
    return torch.flip(tensor, dims=list(dims))


def tta_sliding_window_inference(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    roi_size: tuple[int, int, int],
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    use_amp: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    8-fold flip TTA (x/y/z flips) with softmax probability averaging.

    Returns averaged probabilities [B, C, D, H, W].
    """
    if device is None:
        device = inputs.device

    # Spatial dims for [B, C, D, H, W]
    spatial_axes = (2, 3, 4)
    flip_combos: Iterable[tuple[int, ...]] = [()] + [
        (a,) for a in spatial_axes
    ] + [
        (a, b) for a, b in ((2, 3), (2, 4), (3, 4))
    ] + [
        (2, 3, 4),
    ]

    probs_sum: torch.Tensor | None = None
    n = 0

    model.eval()
    with torch.no_grad():
        for dims in flip_combos:
            x_aug = _flip_spatial_dims(inputs, dims)
            with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = sliding_window_inference(
                    inputs=x_aug,
                    roi_size=tuple(int(v) for v in roi_size),
                    sw_batch_size=int(sw_batch_size),
                    predictor=model,
                    overlap=float(overlap),
                )
            logits = _flip_spatial_dims(logits, dims)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
            n += 1

    assert probs_sum is not None
    return probs_sum / float(max(1, n))

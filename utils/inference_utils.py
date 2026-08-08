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
    accumulate_on_cpu: bool = True,
) -> torch.Tensor:
    """
    8-fold flip TTA (identity, x, y, z, xy, xz, yz, xyz) with softmax averaging.

    Processes ONE augmentation at a time. Softmax accumulation defaults to CPU
    so all 8 full-volume GPU predictions are never retained simultaneously.

    Returns averaged probabilities [B, C, D, H, W] on the same device as `inputs`
    (moved back from CPU if accumulate_on_cpu=True).

    NOTE: TTA is for final inference/evaluation only — not training validation.
    """
    if device is None:
        device = inputs.device

    # Force memory-safe sliding-window batching
    sw_batch_size = 1

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
    with torch.inference_mode():
        for dims in flip_combos:
            x_aug = _flip_spatial_dims(inputs, dims)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = sliding_window_inference(
                    inputs=x_aug,
                    roi_size=tuple(int(v) for v in roi_size),
                    sw_batch_size=int(sw_batch_size),
                    predictor=model,
                    overlap=float(overlap),
                )
            logits = _flip_spatial_dims(logits, dims)
            probs = torch.softmax(logits.float(), dim=1)

            # Free GPU logits ASAP; accumulate on CPU when requested
            del logits
            if accumulate_on_cpu:
                probs_cpu = probs.detach().cpu()
                del probs
                probs_sum = probs_cpu if probs_sum is None else probs_sum + probs_cpu
                del probs_cpu
            else:
                probs_sum = probs if probs_sum is None else probs_sum + probs

            n += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()  # only between TTA views (not every train iter)

    assert probs_sum is not None
    avg = probs_sum / float(max(1, n))
    if accumulate_on_cpu and avg.device.type == "cpu" and device.type == "cuda":
        avg = avg.to(device)
    return avg


from __future__ import annotations

import torch


def multiclass_dice_score_3d(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> float:
    if logits.ndim != 5:
        raise ValueError(f"Expected logits shape [B, C, D, H, W], got {tuple(logits.shape)}")
    if target.ndim != 4:
        raise ValueError(f"Expected target shape [B, D, H, W], got {tuple(target.shape)}")
    if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError(
            f"Shape mismatch: logits {tuple(logits.shape)} vs target {tuple(target.shape)}"
        )

    num_classes = int(logits.shape[1])
    if num_classes <= 1:
        raise ValueError(f"Expected num_classes >= 2, got {num_classes}")

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    dice_per_class = []
    dims = (1, 2, 3)  # D, H, W
    for c in range(1, num_classes):
        pred_c = (preds == c).to(dtype=logits.dtype)
        target_c = (target == c).to(dtype=logits.dtype)

        intersection = (pred_c * target_c).sum(dim=dims)
        pred_sum = pred_c.sum(dim=dims)
        target_sum = target_c.sum(dim=dims)

        dice_c = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice_c = dice_c.mean()
        dice_per_class.append(dice_c)

    mean_dice = torch.stack(dice_per_class).mean()
    return float(mean_dice.detach().item())

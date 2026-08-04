from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


RegionName = Literal["wt", "tc", "et"]


@dataclass(frozen=True)
class BraTSRegionDice:
    """BraTS region Dice scores in [0, 1]."""

    wt: float
    tc: float
    et: float

    @property
    def mean(self) -> float:
        return float((self.wt + self.tc + self.et) / 3.0)


def _to_class_map(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure integer class map [B, D, H, W] or [D, H, W]."""
    if tensor.ndim == 5 and int(tensor.shape[1]) == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim == 4:
        return tensor.long()
    if tensor.ndim == 3:
        return tensor.long()
    raise ValueError(f"Expected 3D/4D label map, got shape {tuple(tensor.shape)}")


def _region_masks(class_map: torch.Tensor, region: RegionName) -> torch.Tensor:
    """Return binary region mask with same shape as class_map."""
    if region == "wt":
        return (class_map == 1) | (class_map == 2) | (class_map == 3)
    if region == "tc":
        return (class_map == 1) | (class_map == 3)
    if region == "et":
        return class_map == 3
    raise ValueError(f"Unknown region: {region}")


def _binary_dice_safe(
    pred_bin: torch.Tensor,
    gt_bin: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Per-sample Dice with BraTS ET absence rule.

    If both pred and gt are empty for the region -> Dice = 1.0.
    """
    pred_bin = pred_bin.bool()
    gt_bin = gt_bin.bool()

    dims = tuple(range(1, pred_bin.ndim))  # spatial dims
    intersection = (pred_bin & gt_bin).sum(dim=dims).float()
    pred_sum = pred_bin.sum(dim=dims).float()
    gt_sum = gt_bin.sum(dim=dims).float()

    both_empty = (pred_sum == 0) & (gt_sum == 0)
    dice = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    return torch.nan_to_num(dice, nan=0.0, posinf=1.0, neginf=0.0)


def logits_to_prediction(logits: torch.Tensor) -> torch.Tensor:
    """Argmax class map from logits [B, C, D, H, W]."""
    return torch.argmax(torch.softmax(logits, dim=1), dim=1)


def compute_region_dice(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    smooth: float = 1.0,
    from_logits: bool = False,
) -> BraTSRegionDice:
    """
    Compute WT / TC / ET Dice.

    Args:
        pred: logits [B,C,D,H,W] or class map [B,D,H,W]
        gt: class map [B,D,H,W] with labels in {0,1,2,3}
        from_logits: if True, `pred` is treated as logits
    """
    gt_map = _to_class_map(gt)
    pred_map = logits_to_prediction(pred) if from_logits else _to_class_map(pred)

    if pred_map.shape != gt_map.shape:
        raise ValueError(f"Shape mismatch pred={tuple(pred_map.shape)} gt={tuple(gt_map.shape)}")

    wt_scores: list[float] = []
    tc_scores: list[float] = []
    et_scores: list[float] = []

    for region, out_list in [("wt", wt_scores), ("tc", tc_scores), ("et", et_scores)]:
        p = _region_masks(pred_map, region)
        g = _region_masks(gt_map, region)
        dice_b = _binary_dice_safe(p, g, smooth=smooth)
        out_list.extend([float(v) for v in dice_b.detach().cpu().tolist()])

    def _mean(xs: list[float]) -> float:
        if not xs:
            return 0.0
        return float(sum(xs) / len(xs))

    return BraTSRegionDice(wt=_mean(wt_scores), tc=_mean(tc_scores), et=_mean(et_scores))

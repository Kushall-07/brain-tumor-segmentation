from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


INCOMPATIBLE_3CLASS_MSG = (
    "Old 3-class checkpoints are incompatible with 4-class model. "
    "Retrain with NUM_CLASSES=4 or use a 4-class checkpoint."
)


def _infer_out_channels(state_dict: Mapping[str, Any]) -> Optional[int]:
    for key in ("out.conv.weight", "out.conv.bias", "output_block.conv.conv.weight"):
        if key in state_dict:
            w = state_dict[key]
            if hasattr(w, "shape") and len(w.shape) >= 1:
                return int(w.shape[0])
    for key, tensor in state_dict.items():
        if key.endswith("out.conv.weight") and hasattr(tensor, "shape"):
            return int(tensor.shape[0])
    return None


def validate_checkpoint_classes(
    checkpoint_path: Path,
    expected_classes: int,
) -> None:
    """Raise if checkpoint output head does not match expected class count."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if not isinstance(state, dict):
        return

    out_ch = _infer_out_channels(state)
    if out_ch is None:
        return
    if int(out_ch) != int(expected_classes):
        if int(out_ch) == 3 and int(expected_classes) == 4:
            raise ValueError(INCOMPATIBLE_3CLASS_MSG)
        raise ValueError(
            f"Checkpoint classes ({out_ch}) incompatible with model ({expected_classes})."
        )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    best_scores: Mapping[str, float],
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "best_scores": {str(k): float(v) for k, v in best_scores.items()},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        try:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass

    torch.save(payload, str(path))

    if metadata is not None:
        meta_path = path.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(dict(metadata), f, indent=2, sort_keys=True)


def try_resume(
    last_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    expected_classes: int,
) -> tuple[int, dict[str, float]]:
    """
    Resume from last.pt if present.

    Returns:
        (start_epoch, best_scores)
    """
    if not last_path.exists():
        return 1, {"mean_dice": float("-inf"), "wt_dice": float("-inf"), "tc_dice": float("-inf"), "et_dice": float("-inf")}

    validate_checkpoint_classes(last_path, expected_classes)
    ckpt = torch.load(str(last_path), map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = int(ckpt.get("epoch", 0))
    best_scores = dict(ckpt.get("best_scores", {}))
    print(f"Resuming from epoch {epoch}")
    return epoch + 1, best_scores

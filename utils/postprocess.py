"""Optional inference-time connected-component post-processing."""

from __future__ import annotations

from typing import Optional

import numpy as np


def remove_small_components(
    pred: np.ndarray,
    *,
    min_size: int = 50,
    classes: tuple[int, ...] = (1, 2, 3),
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Conservatively remove small connected components per class.

    Does NOT keep-only-largest (small ET foci are clinically valid).
    Operates on a copy; raw predictions remain untouched by the caller.

    Returns:
        (postprocessed_mask, stats)
    """
    from scipy import ndimage

    out = np.array(pred, copy=True)
    stats: dict[str, int] = {"removed_components": 0, "removed_voxels": 0}

    for cid in classes:
        binary = out == int(cid)
        if not np.any(binary):
            continue
        labeled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            comp = labeled == i
            size = int(comp.sum())
            if size < int(min_size):
                out[comp] = 0
                stats["removed_components"] += 1
                stats["removed_voxels"] += size

    return out, stats


def maybe_postprocess(
    pred: np.ndarray,
    *,
    enabled: bool,
    min_size: int = 50,
) -> tuple[np.ndarray, Optional[dict[str, int]]]:
    """Apply CC post-processing when enabled; otherwise return pred unchanged."""
    if not enabled:
        print("[postprocess] Connected-component post-processing applied: NO")
        return pred, None
    cleaned, stats = remove_small_components(pred, min_size=min_size)
    print(
        f"[postprocess] Connected-component post-processing applied: YES | "
        f"min_size={min_size} | removed_components={stats['removed_components']} | "
        f"removed_voxels={stats['removed_voxels']}"
    )
    return cleaned, stats

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _load_nifti_to_zyx(path: str | Path, dtype: np.dtype) -> np.ndarray:
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.get_fdata(dtype=dtype)

    # Common cases:
    # - 3D volumes: [X, Y, Z] -> transpose -> [Z, X, Y] == [D,H,W] convention in this repo
    # - 4D volumes: [X, Y, Z, C] -> transpose -> [Z, X, Y, C]
    if data.ndim == 3:
        return np.transpose(data, (2, 0, 1))
    if data.ndim == 4:
        return np.transpose(data, (2, 0, 1, 3))
    raise ValueError(f"Unsupported NIfTI ndim={data.ndim} for {path}")


def _to_uint8_grayscale(vol_zyx: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    v = np.asarray(vol_zyx, dtype=np.float32)
    lo = float(np.percentile(v, pmin))
    hi = float(np.percentile(v, pmax))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(v.min()), float(v.max()) if float(v.max()) > float(v.min()) else (0.0, 1.0)
    v = np.clip((v - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (v * 255.0).astype(np.uint8)


def _select_slice_indices(
    mask_zyx: Optional[np.ndarray],
    depth: int,
    num_slices: int,
    strategy: str = "tumor_then_uniform",
) -> list[int]:
    num_slices = int(num_slices)
    if num_slices <= 0:
        raise ValueError("num_slices must be > 0")

    if depth <= 0:
        return [0]

    if mask_zyx is not None and strategy == "tumor_then_uniform":
        z_fg = np.where(np.any(mask_zyx > 0, axis=(1, 2)))[0]
        if z_fg.size > 0:
            z0, z1 = int(z_fg.min()), int(z_fg.max())
            if z1 > z0:
                zs = np.linspace(z0, z1, num=min(num_slices, (z1 - z0 + 1)), dtype=int).tolist()
                if len(zs) >= num_slices:
                    return zs[:num_slices]
                # pad remaining slices uniformly across full volume
                rest = np.linspace(0, depth - 1, num=num_slices - len(zs), dtype=int).tolist()
                return sorted(set(zs + rest))[:num_slices]

    # fallback: uniform sampling
    return np.linspace(0, depth - 1, num=num_slices, dtype=int).tolist()


@dataclass(frozen=True)
class AxialComparisonSpec:
    num_slices: int = 12
    cols: int = 4
    mri_channel: int = 0
    overlay_alpha: float = 0.45
    title: str | None = None


def save_axial_comparison(
    mri_path: str | Path,
    gt_mask_path: str | Path,
    pred_mask_path: str | Path,
    out_dir: str | Path,
    out_name: str | None = None,
    spec: AxialComparisonSpec = AxialComparisonSpec(),
) -> Path:
    """
    Save a grid of axial slices (MRI / GT / Pred) using matplotlib.

    Conventions:
    - MRI is assumed to be a NIfTI volume (3D or 4D). If 4D, `spec.mri_channel` is used.
    - Masks are assumed to be label maps with integer classes.
    - Internally uses [D,H,W] = [Z,X,Y] convention consistent with `datasets/brats_dataset.py`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mri = _load_nifti_to_zyx(mri_path, dtype=np.float32)
    if mri.ndim == 4:
        c = int(spec.mri_channel)
        if c < 0 or c >= int(mri.shape[-1]):
            raise ValueError(f"mri_channel={c} out of range for MRI with C={int(mri.shape[-1])}")
        mri = mri[..., c]
    if mri.ndim != 3:
        raise ValueError(f"Expected MRI 3D after channel select, got {mri.ndim}D")

    gt = _load_nifti_to_zyx(gt_mask_path, dtype=np.float32)
    pred = _load_nifti_to_zyx(pred_mask_path, dtype=np.float32)

    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError(f"Expected GT and pred to be 3D, got gt={gt.ndim}D pred={pred.ndim}D")

    if gt.shape != mri.shape:
        raise ValueError(f"GT shape {gt.shape} does not match MRI shape {mri.shape}")
    if pred.shape != mri.shape:
        raise ValueError(f"Pred shape {pred.shape} does not match MRI shape {mri.shape}")

    gt_i = gt.astype(np.int16, copy=False)
    pred_i = pred.astype(np.int16, copy=False)

    depth = int(mri.shape[0])
    slice_idxs = _select_slice_indices(gt_i, depth=depth, num_slices=int(spec.num_slices))

    mri_u8 = _to_uint8_grayscale(mri)

    # Headless-safe backend
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Discrete colors for BraTS 4-class: 0 bg, 1 NCR/NET red, 2 edema green, 3 ET gold
    cmap = ListedColormap(["black", "#e41a1c", "#4daf4a", "#ffd700"])

    n = len(slice_idxs)
    cols = max(1, int(spec.cols))
    rows = int(np.ceil(n / cols))

    fig_w = 3.6 * cols
    fig_h = 3.2 * rows
    fig, axes = plt.subplots(rows, cols * 3, figsize=(fig_w * 3, fig_h), dpi=200)
    axes = np.atleast_2d(axes)

    def _panel(ax, mri_slice: np.ndarray, mask_slice: Optional[np.ndarray], title: str) -> None:
        ax.imshow(mri_slice, cmap="gray", interpolation="nearest")
        if mask_slice is not None:
            ax.imshow(mask_slice, cmap=cmap, interpolation="nearest", alpha=float(spec.overlay_alpha), vmin=0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, z in enumerate(slice_idxs):
        r = i // cols
        c = i % cols
        base = c * 3

        mri_sl = mri_u8[z]
        gt_sl = gt_i[z]
        pred_sl = pred_i[z]

        _panel(axes[r, base + 0], mri_sl, None, f"MRI (z={z})")
        _panel(axes[r, base + 1], mri_sl, gt_sl, "GT")
        _panel(axes[r, base + 2], mri_sl, pred_sl, "Pred")

    # Hide unused axes (if grid not full)
    total_cells = rows * cols
    for i in range(n, total_cells):
        r = i // cols
        c = i % cols
        base = c * 3
        for j in range(3):
            axes[r, base + j].axis("off")

    if spec.title:
        fig.suptitle(spec.title, fontsize=12, y=0.995)

    fig.tight_layout()

    if out_name is None:
        out_name = f"{Path(mri_path).stem}_comparison.png"
        if out_name.endswith(".nii_comparison.png"):
            out_name = out_name.replace(".nii_comparison.png", "_comparison.png")
        if out_name.endswith(".nii.gz_comparison.png"):
            out_name = out_name.replace(".nii.gz_comparison.png", "_comparison.png")

    out_path = out_dir / out_name
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


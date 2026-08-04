from __future__ import annotations

import argparse
from pathlib import Path

from utils.select_visual_cases import _dice_binary  # type: ignore
from utils.visualization import AxialComparisonSpec, save_axial_comparison

import nibabel as nib
import numpy as np


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata())


def _rank_cases(val_root: Path, pred_root: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for case_dir in sorted([p for p in val_root.iterdir() if p.is_dir()]):
        case_id = case_dir.name
        gt_path = case_dir / f"{case_id}-seg.nii"
        if not gt_path.exists():
            gt_path = case_dir / f"{case_id}-seg.nii.gz"
        pred_path = pred_root / f"{case_id}_pred.nii.gz"
        if not (gt_path.exists() and pred_path.exists()):
            continue
        gt = _load_mask(gt_path)
        pr = _load_mask(pred_path)
        rows.append((_dice_binary(pr, gt), case_id))
    rows.sort(key=lambda x: x[0])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative visualizations for selected cases")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation root directory")
    parser.add_argument("--pred_dir", type=str, required=True, help="Predictions directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for PNGs")
    parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name for plot titles")
    parser.add_argument("--mri_modality", type=str, default="t1c", help="MRI modality filename suffix (e.g. t1c)")
    parser.add_argument("--bottom", type=int, default=2)
    parser.add_argument("--middle", type=int, default=2)
    parser.add_argument("--top", type=int, default=2)
    parser.add_argument("--num_slices", type=int, default=12)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    val_root = Path(args.val_dir)
    pred_root = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    rows = _rank_cases(val_root=val_root, pred_root=pred_root)
    if not rows:
        raise SystemExit("No cases found (check paths).")

    picks: list[str] = []
    picks += [cid for _d, cid in rows[: min(len(rows), int(args.bottom))]]
    if int(args.middle) > 0 and len(rows) >= 3:
        m = len(rows) // 2
        half = int(args.middle) // 2
        lo = max(0, m - half)
        hi = min(len(rows), lo + int(args.middle))
        picks += [cid for _d, cid in rows[lo:hi]]
    picks += [cid for _d, cid in rows[-min(len(rows), int(args.top)) :]]

    # unique preserve order
    selected: list[str] = []
    for cid in picks:
        if cid not in selected:
            selected.append(cid)

    spec = AxialComparisonSpec(
        num_slices=int(args.num_slices),
        cols=int(args.cols),
        mri_channel=0,
        overlay_alpha=float(args.alpha),
        title=None,
    )

    score_map = {cid: d for d, cid in rows}
    print("Selected cases (tumor-vs-background Dice):")
    for cid in selected:
        print(f"- {cid}: {score_map[cid]:.4f}")

    for cid in selected:
        case_dir = val_root / cid
        mri_path = case_dir / f"{cid}-{args.mri_modality}.nii"
        if not mri_path.exists():
            mri_path = case_dir / f"{cid}-{args.mri_modality}.nii.gz"
        gt_path = case_dir / f"{cid}-seg.nii"
        if not gt_path.exists():
            gt_path = case_dir / f"{cid}-seg.nii.gz"
        pred_path = pred_root / f"{cid}_pred.nii.gz"

        title = f"{args.exp_name} | {cid} ({args.mri_modality.upper()}) | dice={score_map[cid]:.3f}"
        out_path = save_axial_comparison(
            mri_path=mri_path,
            gt_mask_path=gt_path,
            pred_mask_path=pred_path,
            out_dir=out_dir,
            out_name=None,
            spec=AxialComparisonSpec(
                num_slices=spec.num_slices,
                cols=spec.cols,
                mri_channel=spec.mri_channel,
                overlay_alpha=spec.overlay_alpha,
                title=title,
            ),
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


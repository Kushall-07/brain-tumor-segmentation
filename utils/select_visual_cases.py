from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata())


def _dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best/average/worst cases for qualitative visualization")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation root directory (contains case subfolders)")
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Predictions directory (contains <case_id>_pred.nii.gz)",
    )
    parser.add_argument("--bottom", type=int, default=2, help="Number of worst cases to pick")
    parser.add_argument("--middle", type=int, default=2, help="Number of middle (median-ish) cases to pick")
    parser.add_argument("--top", type=int, default=2, help="Number of best cases to pick")
    args = parser.parse_args()

    val_root = Path(args.val_dir)
    pred_root = Path(args.pred_dir)

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
        dice = _dice_binary(pr, gt)
        rows.append((dice, case_id))

    rows.sort(key=lambda x: x[0])
    if not rows:
        raise SystemExit("No cases found (check --val_dir and --pred_dir paths).")

    picks: list[str] = []
    bottom_n = min(int(args.bottom), len(rows))
    picks += [cid for _d, cid in rows[:bottom_n]]

    if int(args.middle) > 0 and len(rows) >= 3:
        m = len(rows) // 2
        half = int(args.middle) // 2
        lo = max(0, m - half)
        hi = min(len(rows), lo + int(args.middle))
        picks += [cid for _d, cid in rows[lo:hi]]

    top_n = min(int(args.top), len(rows))
    picks += [cid for _d, cid in rows[-top_n:]]

    # Unique, preserve order
    out: list[str] = []
    for c in picks:
        if c not in out:
            out.append(c)

    score_map = {cid: d for d, cid in rows}
    for cid in out:
        print(f"{cid}\t{score_map[cid]:.6f}")


if __name__ == "__main__":
    main()


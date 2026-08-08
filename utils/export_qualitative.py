"""
Export best / median / worst qualitative cases (axial + optional 3D HTML) for IEEE-style figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from configs.config import Config, PATCH_SIZE
from models.model_factory import build_model
from utils.brats_metrics import compute_region_dice
from utils.checkpoint_utils import validate_checkpoint_classes
from utils.inference_utils import tta_sliding_window_inference
from utils.visualization import AxialComparisonSpec, save_axial_comparison

try:
    from inference.visualize_3d import run_visualization, VisualizationConfig
except ImportError:
    run_visualization = None  # type: ignore
    VisualizationConfig = None  # type: ignore


def _find_seg(case_dir: Path, case_id: str) -> Path:
    for name in (f"{case_id}-seg.nii.gz", f"{case_id}-seg.nii"):
        p = case_dir / name
        if p.exists():
            return p
    matches = sorted(case_dir.glob("*seg*.nii*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No segmentation file in {case_dir}")


def _find_mri(case_dir: Path, case_id: str, token: str) -> Path:
    for name in (f"{case_id}-{token}.nii.gz", f"{case_id}-{token}.nii"):
        p = case_dir / name
        if p.exists():
            return p
    matches = sorted(case_dir.glob(f"*{token}*.nii*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No MRI token '{token}' in {case_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export qualitative best/middle/worst cases")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mri_token", type=str, default="t2f")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--bottom", type=int, default=3)
    parser.add_argument("--middle", type=int, default=2)
    parser.add_argument("--export_3d", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    repo = Path(__file__).resolve().parents[1]
    exp_dir = repo / "outputs" / str(args.exp_name)
    ckpt = Path(args.checkpoint) if args.checkpoint else exp_dir / "checkpoints" / "best_mean_dice.pt"
    if not ckpt.exists():
        ckpt = exp_dir / "checkpoints" / "best.pt"

    qual_dir = exp_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_checkpoint_classes(ckpt, cfg.num_classes)

    from datasets.brats_dataset import BraTSDataset

    ds = BraTSDataset(root_dir=args.data_dir)
    model, _report = build_model(
        cfg.model_name,
        in_channels=cfg.input_channels,
        out_channels=cfg.num_classes,
        patch_size=cfg.patch_size,
        baseline_features=cfg.baseline_unet_features,
        residual_features=cfg.residual_unet_features,
        swin_feature_size=cfg.swin_feature_size,
        swin_use_checkpoint=cfg.swin_use_checkpoint,
    )
    model = model.to(device)

    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    rows: list[tuple[float, str, int]] = []
    for i in range(len(ds)):
        image, mask, case_id = ds[int(i)]
        x = image.unsqueeze(0).to(device)
        y = mask.unsqueeze(0).to(device)
        with torch.no_grad():
            if cfg.use_tta:
                probs = tta_sliding_window_inference(model, x, roi_size=cfg.patch_size, device=device)
                logits = torch.log(probs + 1e-8)
            else:
                logits = tta_sliding_window_inference(model, x, roi_size=cfg.patch_size, device=device)  # noqa: using tta path always for ranking quality
        scores = compute_region_dice(logits, y, from_logits=True)
        rows.append((scores.mean, str(case_id), int(i)))

    rows.sort(key=lambda t: t[0])
    picks: list[tuple[str, float, str]] = []
    for d, cid, _ in rows[: int(args.bottom)]:
        picks.append(("worst", d, cid))
    if rows:
        m = len(rows) // 2
        half = max(1, int(args.middle) // 2)
        for d, cid, _ in rows[max(0, m - half) : min(len(rows), m + half)]:
            picks.append(("median", d, cid))
    for d, cid, _ in rows[-int(args.top) :]:
        picks.append(("best", d, cid))

    seen: set[str] = set()
    spec = AxialComparisonSpec(num_slices=12, cols=4, overlay_alpha=0.45)

    for tag, mean_dice, case_id in picks:
        if case_id in seen:
            continue
        seen.add(case_id)

        case_dir = Path(args.data_dir) / case_id
        mri_path = _find_mri(case_dir, case_id, args.mri_token)
        gt_path = _find_seg(case_dir, case_id)

        pred_path = qual_dir / f"{tag}_{case_id}_pred.nii.gz"
        idx = next(i for _, cid, i in rows if cid == case_id)
        image, _mask, _ = ds[int(idx)]
        x = image.unsqueeze(0).to(device)
        with torch.no_grad():
            probs = tta_sliding_window_inference(model, x, roi_size=cfg.patch_size, device=device)
            pred = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        ref = nib.as_closest_canonical(nib.load(str(sorted(case_dir.glob("*.nii*"))[0])))
        pred_xyz = np.transpose(pred, (1, 2, 0))
        nib.save(nib.Nifti1Image(pred_xyz, ref.affine), str(pred_path))

        out_name = f"{tag}_{case_id}_mean{mean_dice:.3f}.png"
        save_axial_comparison(
            mri_path=mri_path,
            gt_mask_path=gt_path,
            pred_mask_path=pred_path,
            out_dir=qual_dir,
            out_name=out_name,
            spec=AxialComparisonSpec(
                num_slices=spec.num_slices,
                cols=spec.cols,
                overlay_alpha=spec.overlay_alpha,
                title=f"{tag.upper()} | {case_id} | Mean Dice={mean_dice:.3f}",
            ),
        )
        print(f"Saved {tag}: {case_id} mean_dice={mean_dice:.4f}")

        if args.export_3d and run_visualization is not None and VisualizationConfig is not None:
            run_visualization(
                mri_path=mri_path,
                pred_path=pred_path,
                out_dir=qual_dir / "3d",
                gt_path=gt_path,
                out_stem=f"{tag}_{case_id}",
                cfg=VisualizationConfig(title=f"{tag} | {case_id} | WT/TC/ET"),
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from configs.config import PATCH_SIZE
from datasets.brats_dataset import BraTSDataset
from models.model_factory import build_model
from utils.brats_metrics import compute_region_dice
from utils.visualization import AxialComparisonSpec, save_axial_comparison


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)


def _find_case_file(case_dir: Path, token: str) -> Path:
    matches = sorted(case_dir.glob(f"*{token}*.nii.gz")) + sorted(case_dir.glob(f"*{token}*.nii"))
    if len(matches) == 0:
        raise FileNotFoundError(f"Could not find file containing '{token}' in {case_dir}")
    if len(matches) > 1:
        raise FileExistsError(f"Multiple files containing '{token}' in {case_dir}: {[m.name for m in matches]}")
    return matches[0]


def _reference_nifti(case_dir: Path) -> nib.Nifti1Image:
    candidates = sorted(case_dir.glob("*.nii.gz")) + sorted(case_dir.glob("*.nii"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No NIfTI files found in: {case_dir}")
    return nib.as_closest_canonical(nib.load(str(candidates[0])))


def _save_pred_mask(pred_zyx: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    ref_shape = tuple(int(v) for v in ref_img.shape[:3])
    if tuple(int(v) for v in pred_zyx.shape) == ref_shape:
        pred_out = pred_zyx
    elif tuple(int(v) for v in np.transpose(pred_zyx, (1, 2, 0)).shape) == ref_shape:
        pred_out = np.transpose(pred_zyx, (1, 2, 0))
    else:
        raise ValueError(f"Prediction shape {pred_zyx.shape} does not match reference shape {ref_shape}")

    out_img = nib.Nifti1Image(pred_out.astype(np.uint8, copy=False), affine=ref_img.affine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))


@torch.no_grad()
def _predict_logits_full_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_size: Tuple[int, int, int] = PATCH_SIZE,
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    use_amp: bool = True,
) -> torch.Tensor:
    x = image.unsqueeze(0).to(device)  # [1, 4, D, H, W]

    with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
        logits = sliding_window_inference(
            inputs=x,
            roi_size=tuple(int(v) for v in roi_size),
            sw_batch_size=int(sw_batch_size),
            predictor=model,
            overlap=float(overlap),
        )
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PNG visualizations for best-Dice validation cases")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing case subfolders (e.g. BraTS/Validation)")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name under outputs/<exp_name>/")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output dir for PNGs. If omitted uses outputs/<exp_name>/visualizations",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of best Dice cases to visualize",
    )
    parser.add_argument(
        "--mri_token",
        type=str,
        default="t2f",
        help="Which modality token to use as the background MRI for visualization (e.g. t2f, t1n)",
    )
    parser.add_argument("--num_slices", type=int, default=12)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = repo_root / "outputs"

    exp_dir = outputs_root / str(args.exp_name)
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        legacy_best = outputs_root / "checkpoints" / "best.pt"
        if legacy_best.exists():
            print(f"Warning: checkpoint not found at {checkpoint_path}. Falling back to legacy checkpoint: {legacy_best}")
            checkpoint_path = legacy_best
        else:
            raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else (exp_dir / "visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()

    # Model config is stored in the checkpoint JSON, but training already uses config.py.
    # We keep it simple: infer architecture from checkpoint metadata if present later.
    # For now, build residual_unet-compatible default; user can swap via configs/config.py if needed.
    # The build_model call requires a model_name and widths; we can read from exp config via configs.Config.
    from configs.config import Config

    cfg = Config()
    model = build_model(
        cfg.model_name,
        in_channels=cfg.input_channels,
        out_channels=cfg.num_classes,
        patch_size=cfg.patch_size,
        baseline_features=cfg.baseline_unet_features,
        residual_features=cfg.residual_unet_features,
        swin_feature_size=cfg.swin_feature_size,
        swin_use_checkpoint=cfg.swin_use_checkpoint,
    ).to(device)

    _load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    model.eval()

    ds = BraTSDataset(root_dir=args.data_dir)

    scores: List[Tuple[str, float, int]] = []
    print(f"Evaluating {len(ds)} cases for best Dice...")

    for i in range(len(ds)):
        image, mask, case_id = ds[int(i)]
        logits = _predict_logits_full_volume(model=model, image=image, device=device, roi_size=PATCH_SIZE, overlap=0.5)

        y = mask.unsqueeze(0).to(device)  # [1, D, H, W]
        region = compute_region_dice(logits, y, from_logits=True)
        dice = region.mean
        scores.append((str(case_id), float(dice), int(i)))
        print(
            f"[{i+1}/{len(ds)}] {case_id} mean={dice:.4f} "
            f"WT={region.wt:.4f} TC={region.tc:.4f} ET={region.et:.4f}"
        )

    scores.sort(key=lambda t: t[1], reverse=True)
    top = scores[: max(0, int(args.k))]

    print("\nTop cases:")
    for rank, (case_id, dice, _idx) in enumerate(top, start=1):
        print(f"{rank:02d}. {case_id} dice={dice:.6f}")

    spec = AxialComparisonSpec(
        num_slices=int(args.num_slices),
        cols=int(args.cols),
        mri_channel=0,
        overlay_alpha=float(args.alpha),
        title=None,
    )

    preds_dir = exp_dir / "predictions" / "val_best_cases"
    preds_dir.mkdir(parents=True, exist_ok=True)

    for rank, (case_id, dice, idx) in enumerate(top, start=1):
        image, mask, _case_id2 = ds[int(idx)]
        case_dir = Path(args.data_dir) / str(case_id)

        ref_img = _reference_nifti(case_dir)

        logits = _predict_logits_full_volume(model=model, image=image, device=device, roi_size=PATCH_SIZE, overlap=0.5)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8, copy=False)  # [D,H,W]

        pred_path = preds_dir / f"{case_id}_pred.nii.gz"
        _save_pred_mask(pred, ref_img=ref_img, out_path=pred_path)

        mri_path = _find_case_file(case_dir, str(args.mri_token))
        gt_path = _find_case_file(case_dir, "seg")

        out_name = f"{rank:02d}_{case_id}_dice_{dice:.4f}.png"
        save_axial_comparison(
            mri_path=mri_path,
            gt_mask_path=gt_path,
            pred_mask_path=pred_path,
            out_dir=out_dir,
            out_name=out_name,
            spec=spec,
        )

        print(f"Saved: {out_dir / out_name}")


if __name__ == "__main__":
    main()

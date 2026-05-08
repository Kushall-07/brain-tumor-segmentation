
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from configs.config import INPUT_CHANNELS, NUM_CLASSES, PATCH_SIZE
from datasets.brats_dataset import BraTSDataset
from models.unet3d import UNet3D


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    in_channels: int = INPUT_CHANNELS,
    out_channels: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
) -> nn.Module:
    model = UNet3D(in_channels=int(in_channels), out_channels=int(out_channels))
    if device is not None:
        model = model.to(device)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)


def load_single_case(data_dir: str | Path, case_index: int = 0) -> Tuple[torch.Tensor, str, Path]:
    ds = BraTSDataset(root_dir=data_dir)
    image, _mask, case_id = ds[int(case_index)]

    if image.ndim != 4 or int(image.shape[0]) != 4:
        raise ValueError(f"Expected image shape [4, D, H, W], got: {tuple(image.shape)}")

    case_dir = Path(data_dir) / case_id
    return image, case_id, case_dir


def _case_reference_nii(case_dir: Path) -> nib.Nifti1Image:
    candidates = sorted(case_dir.glob("*.nii.gz")) + sorted(case_dir.glob("*.nii"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No NIfTI files found in case folder: {case_dir}")
    return nib.as_closest_canonical(nib.load(str(candidates[0])))


def run_sliding_window(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_size: tuple[int, int, int] = PATCH_SIZE,
    sw_batch_size: int = 1,
    overlap: float = 0.25,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()

    x = image.unsqueeze(0).to(device)  # [1, 4, D, H, W]

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=x,
            roi_size=tuple(int(v) for v in roi_size),
            sw_batch_size=int(sw_batch_size),
            predictor=model,
            overlap=float(overlap),
        )

        probs = torch.softmax(logits, dim=1)  # [1, C, D, H, W]
        pred = torch.argmax(probs, dim=1)  # [1, D, H, W]

    pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8, copy=False)
    probs_np = probs.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return pred_np, probs_np


def save_nifti_mask(
    pred_zyx: np.ndarray,
    reference_img: nib.Nifti1Image,
    out_path: str | Path,
) -> None:
    if pred_zyx.ndim != 3:
        raise ValueError(f"Expected pred mask [D,H,W], got: {pred_zyx.shape}")

    ref_shape = tuple(int(v) for v in reference_img.shape[:3])
    if tuple(int(v) for v in pred_zyx.shape) == ref_shape:
        pred_out = pred_zyx
    elif tuple(int(v) for v in np.transpose(pred_zyx, (1, 2, 0)).shape) == ref_shape:
        pred_out = np.transpose(pred_zyx, (1, 2, 0))
    else:
        raise ValueError(f"Prediction shape {pred_zyx.shape} does not match reference shape {ref_shape}")

    out_img = nib.Nifti1Image(pred_out.astype(np.uint8, copy=False), affine=reference_img.affine)
    nib.save(out_img, str(out_path))


def save_nifti_probabilities(
    probs_czyx: np.ndarray,
    reference_img: nib.Nifti1Image,
    out_path: str | Path,
) -> None:
    if probs_czyx.ndim != 4:
        raise ValueError(f"Expected probs [C,D,H,W], got: {probs_czyx.shape}")

    probs_zyxc = np.moveaxis(probs_czyx, 0, -1)

    ref_shape = tuple(int(v) for v in reference_img.shape[:3])
    if tuple(int(v) for v in probs_zyxc.shape[:3]) == ref_shape:
        probs_out = probs_zyxc
    elif tuple(int(v) for v in np.transpose(probs_zyxc, (1, 2, 0, 3)).shape[:3]) == ref_shape:
        probs_out = np.transpose(probs_zyxc, (1, 2, 0, 3))
    else:
        raise ValueError(f"Probabilities shape {probs_zyxc.shape} does not match reference shape {ref_shape}")

    out_img = nib.Nifti1Image(probs_out.astype(np.float32, copy=False), affine=reference_img.affine)
    nib.save(out_img, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Brain Tumor Segmentation Inference (BraTS)")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing case subfolders")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Path to trained checkpoint (.pt)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--case_index", type=int, default=0, help="Index of case in data_dir to run inference on")
    parser.add_argument("--save_probs", action="store_true", help="If set, save class probabilities as NIfTI")
    args = parser.parse_args()

    device = get_device()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(device=device)
    load_checkpoint(model=model, checkpoint_path=args.checkpoint, device=device)

    image, case_id, case_dir = load_single_case(args.data_dir, case_index=args.case_index)
    ref_img = _case_reference_nii(case_dir)

    pred_mask, probs = run_sliding_window(model=model, image=image, device=device, roi_size=PATCH_SIZE)

    mask_path = out_dir / f"{case_id}_pred.nii.gz"
    save_nifti_mask(pred_mask, reference_img=ref_img, out_path=mask_path)

    if args.save_probs:
        probs_path = out_dir / f"{case_id}_probs.nii.gz"
        save_nifti_probabilities(probs, reference_img=ref_img, out_path=probs_path)

    print(f"Saved mask: {mask_path}")
    if args.save_probs:
        print(f"Saved probabilities: {probs_path}")


if __name__ == "__main__":
    main()

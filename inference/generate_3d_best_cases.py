"""Generate 3D interactive visualizations for best validation cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from configs.config import PATCH_SIZE
from datasets.brats_dataset import BraTSDataset
from inference.visualize_3d import VisualizationConfig, create_interactive_figure, build_meshes_from_volumes
from models.model_factory import build_model
from utils.brats_metrics import compute_region_dice
from utils.mesh_utils import load_mask_volume, load_mri_volume


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)


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


def main():
    # Load best cases from metrics
    project_root = Path(__file__).resolve().parents[1]
    exp_dir = project_root / "outputs" / "exp_swinunetr_4class_et_fixed"
    best_cases_path = exp_dir / "metrics" / "best_cases.json"
    
    with open(best_cases_path) as f:
        best_cases = json.load(f)
    
    print(f"Best cases to visualize: {best_cases}")
    
    # Data paths
    data_dir = project_root.parent / "BraTS" / "Validation"
    preds_dir = exp_dir / "predictions" / "val_best_cases"
    preds_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = exp_dir / "visualizations" / "3d_best_cases"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model for generating missing predictions
    from configs.config import Config
    cfg = Config()
    device = _get_device()
    
    checkpoint_path = exp_dir / "checkpoints" / "best_mean_dice.pt"
    if not checkpoint_path.exists():
        # Try alternative checkpoint names
        alternatives = ["best.pt", "best_wt.pt", "best_tc.pt", "best_et.pt"]
        for alt in alternatives:
            alt_path = exp_dir / "checkpoints" / alt
            if alt_path.exists():
                print(f"Using alternative checkpoint: {alt_path}")
                checkpoint_path = alt_path
                break
        if not checkpoint_path.exists():
            raise SystemExit(f"Checkpoint not found: tried best_mean_dice.pt and alternatives in {exp_dir / 'checkpoints'}")
    
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
    _load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    model.eval()
    
    # Load dataset for generating predictions
    ds = BraTSDataset(root_dir=data_dir)
    
    # Visualization config (paper-ready)
    cfg = VisualizationConfig(
        title="3D Brain Tumor Segmentation - Best Cases",
        brain_opacity=0.12,
        tumor_opacity=0.90,
        dark_mode=True,
        show_axes=False,
        camera_preset="default",
        auto_rotate=False,
        downsample_brain=2,
        downsample_tumor=1,
        marching_step_brain=2,
        marching_step_tumor=1,
        crop_margin_vox=20,
        class_ids=(1, 2, 3),  # NCR/NET, Edema, ET
    )
    
    # Optional: Use post-processed predictions to remove floating artifacts
    use_postprocessed = True  # Set to False to use raw predictions
    
    # Generate visualizations for each best case
    for metric, case_id in best_cases.items():
        print(f"\nGenerating 3D visualization for {metric}: {case_id}")
        
        case_dir = data_dir / case_id
        if not case_dir.exists():
            print(f"Case directory not found: {case_dir}")
            continue
        
        # Find MRI modality (prefer t1c for contrast)
        mri_file = None
        for modality in ["t1c", "t1n", "t2w", "t2f"]:
            matches = list(case_dir.glob(f"*{modality}*.nii.gz")) + list(case_dir.glob(f"*{modality}*.nii"))
            if matches:
                mri_file = matches[0]
                break
        
        if not mri_file:
            print(f"No MRI file found for {case_id}")
            continue
        
        # Find or generate prediction mask
        pred_file = preds_dir / f"{case_id}_pred.nii.gz"
        if use_postprocessed:
            pred_file_cleaned = preds_dir / f"{case_id}_pred_cleaned.nii.gz"
            if pred_file_cleaned.exists():
                pred_file = pred_file_cleaned
                print(f"Using post-processed prediction: {pred_file_cleaned.name}")
        
        if not pred_file.exists():
            print(f"Generating prediction for {case_id}...")
            try:
                # Find the dataset index for this case
                case_idx = None
                for i in range(len(ds)):
                    _, _, ds_case_id = ds[int(i)]
                    if ds_case_id == case_id:
                        case_idx = i
                        break
                
                if case_idx is None:
                    print(f"Case {case_id} not found in dataset")
                    continue
                
                # Generate prediction
                image, mask, _ = ds[int(case_idx)]
                logits = _predict_logits_full_volume(model=model, image=image, device=device, roi_size=PATCH_SIZE, overlap=0.5)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8, copy=False)  # [D,H,W]
                
                # Save prediction
                ref_img = _reference_nifti(case_dir)
                _save_pred_mask(pred, ref_img=ref_img, out_path=pred_file)
                print(f"Saved prediction: {pred_file}")
                
            except Exception as e:
                print(f"Error generating prediction for {case_id}: {e}")
                continue
        
        try:
            # Load volumes
            mri_vol = load_mri_volume(mri_file)
            pred_vol = load_mask_volume(pred_file)
            
            # Build meshes
            brain_meshes, tumor_meshes = build_meshes_from_volumes(
                mri_vol.array, pred_vol.array, cfg=cfg
            )
            
            # Create interactive figure
            fig = create_interactive_figure(brain_meshes, tumor_meshes, cfg=cfg)
            
            # Update title with metrics
            title = f"Best {metric.replace('_', ' ').title()}: {case_id}"
            if metric == "best_mean":
                title += f"<br>Mean Dice: {best_cases.get('best_mean', 'N/A')}"
            elif metric == "best_wt":
                title += f"<br>WT Dice: {best_cases.get('best_wt', 'N/A')}"
            elif metric == "best_tc":
                title += f"<br>TC Dice: {best_cases.get('best_tc', 'N/A')}"
            elif metric == "best_et":
                title += f"<br>ET Dice: {best_cases.get('best_et', 'N/A')}"
            
            fig.update_layout(title=title)
            
            # Save as HTML
            out_html = viz_dir / f"{metric}_{case_id}.html"
            fig.write_html(str(out_html))
            print(f"Saved 3D visualization: {out_html}")
            
        except Exception as e:
            print(f"Error generating visualization for {case_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nAll 3D visualizations saved to: {viz_dir}")
    print("Open HTML files in a web browser for interactive 3D viewing.")


if __name__ == "__main__":
    main()
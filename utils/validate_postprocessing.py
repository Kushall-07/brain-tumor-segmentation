"""Quantitative validation of connected-component post-processing.

Compares raw vs cleaned predictions against ground truth for all validation cases.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy.ndimage import label

from configs.config import PATCH_SIZE
from datasets.brats_dataset import BraTSDataset
from models.model_factory import build_model
from utils.brats_metrics import compute_region_dice


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


def _save_nifti(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out_img = nib.Nifti1Image(data.astype(np.uint8), affine=ref_img.affine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))


def _load_volume_as_array(nifti_path: Path) -> np.ndarray:
    """Load NIfTI and return [D,H,W] array."""
    img = nib.as_closest_canonical(nib.load(str(nifti_path)))
    data = img.get_fdata()
    data = np.transpose(data, (2, 0, 1))
    return np.rint(data).astype(np.int16)


def calculate_dice_overlap(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate Dice overlap coefficient between two binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    if not gt.any() and not pred.any():
        return 1.0  # Both empty
    
    if not gt.any() or not pred.any():
        return 0.0  # One empty
    
    intersection = np.sum(pred & gt)
    dice = 2.0 * intersection / (np.sum(pred) + np.sum(gt))
    return dice


def calculate_class_overlap(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    """Calculate overlap between specific class in prediction and ground truth."""
    pred_class = (pred == class_id).astype(bool)
    gt_class = (gt == class_id).astype(bool)
    return calculate_dice_overlap(pred_class, gt_class)


def calculate_volume(pred: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> float:
    """Calculate physical volume in mm³."""
    voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    return np.sum(pred) * voxel_volume


def analyze_postprocessing_impact(
    data_dir: Path,
    exp_dir: Path,
    checkpoint_path: Path,
    raw_preds_dir: Path,
    cleaned_preds_dir: Path,
    min_voxel_count: int = 200,
    keep_largest_per_class: bool = True
) -> Tuple[List[Dict], Dict]:
    """Compare raw vs cleaned predictions against ground truth."""
    
    # Load model
    from configs.config import Config
    cfg = Config()
    device = _get_device()
    
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
    
    # Load dataset
    ds = BraTSDataset(root_dir=data_dir)
    
    # Get voxel spacing from first case
    first_case_dir = sorted(data_dir.iterdir())[0]
    sample_mri_files = list(first_case_dir.glob("*.nii.gz")) + list(first_case_dir.glob("*.nii"))
    if not sample_mri_files:
        raise FileNotFoundError(f"No NIfTI files found in: {first_case_dir}")
    sample_mri = sample_mri_files[0]
    sample_img = nib.as_closest_canonical(nib.load(str(sample_mri)))
    voxel_spacing = sample_img.header.get_zooms()
    
    results = []
    
    print(f"Evaluating {len(ds)} validation cases...")
    
    for i in range(len(ds)):
        image, mask, case_id = ds[int(i)]
        case_dir = data_dir / case_id
        
        # Load ground truth
        gt_file = case_dir / f"{case_id}-seg.nii"
        if not gt_file.exists():
            gt_file = list(case_dir.glob("*seg*.nii.gz"))[0]
        
        gt_data = _load_volume_as_array(gt_file)
        
        # Get raw prediction from model
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=image_tensor,
                roi_size=cfg.patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )
        probs = torch.softmax(logits, dim=1)
        raw_pred = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)
        
        # Apply post-processing
        pred_clean = raw_pred.copy()
        for class_id in [1, 2, 3]:
            class_mask = (raw_pred == class_id).astype(np.int16)
            if not np.any(class_mask):
                continue
            
            labeled_array, num_features = label(class_mask)
            component_sizes = []
            for comp_id in range(1, num_features + 1):
                comp_mask = (labeled_array == comp_id)
                voxel_count = np.sum(comp_mask)
                component_sizes.append((comp_id, voxel_count))
            
            component_sizes.sort(key=lambda x: x[1], reverse=True)
            
            if keep_largest_per_class and component_sizes:
                largest_id = component_sizes[0][0]
                keep_mask = (labeled_array == largest_id)
                remove_mask = (labeled_array != largest_id) & (labeled_array > 0)
                pred_clean[remove_mask] = 0
            else:
                for comp_id, voxel_count in component_sizes:
                    if voxel_count < min_voxel_count:
                        remove_mask = (labeled_array == comp_id)
                        pred_clean[remove_mask] = 0
        
        # Calculate metrics for raw prediction
        raw_metrics = compute_region_dice(torch.from_numpy(raw_pred).unsqueeze(0), 
                                           torch.from_numpy(gt_data).unsqueeze(0), 
                                           from_logits=False)
        
        # Calculate metrics for cleaned prediction
        clean_metrics = compute_region_dice(torch.from_numpy(pred_clean).unsqueeze(0),
                                            torch.from_numpy(gt_data).unsqueeze(0),
                                            from_logits=False)
        
        # Calculate volumes
        raw_volumes = {
            'wt': calculate_volume((raw_pred > 0).astype(np.int16), voxel_spacing),
            'tc': calculate_volume(((raw_pred == 1) | (raw_pred == 3)).astype(np.int16), voxel_spacing),
            'et': calculate_volume((raw_pred == 3).astype(np.int16), voxel_spacing)
        }
        
        clean_volumes = {
            'wt': calculate_volume((pred_clean > 0).astype(np.int16), voxel_spacing),
            'tc': calculate_volume(((pred_clean == 1) | (pred_clean == 3)).astype(np.int16), voxel_spacing),
            'et': calculate_volume((pred_clean == 3).astype(np.int16), voxel_spacing)
        }
        
        # Calculate overlap for removed components
        removed_voxels = raw_pred - pred_clean
        total_removed = np.sum(removed_voxels > 0)
        removed_in_gt = np.sum((removed_voxels > 0) & (gt_data > 0))
        removed_in_correct_class = 0
        
        for class_id in [1, 2, 3]:
            class_removed = (removed_voxels == class_id).astype(bool)
            class_gt = (gt_data == class_id).astype(bool)
            removed_in_correct_class += np.sum(class_removed & class_gt)
        
        case_result = {
            'case_id': str(case_id),
            'raw_wt_dice': float(raw_metrics.wt),
            'clean_wt_dice': float(clean_metrics.wt),
            'raw_tc_dice': float(raw_metrics.tc),
            'clean_tc_dice': float(clean_metrics.tc),
            'raw_et_dice': float(raw_metrics.et),
            'clean_et_dice': float(clean_metrics.et),
            'raw_mean_dice': float(raw_metrics.mean),
            'clean_mean_dice': float(clean_metrics.mean),
            'raw_wt_volume': float(raw_volumes['wt']),
            'clean_wt_volume': float(clean_volumes['wt']),
            'raw_tc_volume': float(raw_volumes['tc']),
            'clean_tc_volume': float(clean_volumes['tc']),
            'raw_et_volume': float(raw_volumes['et']),
            'clean_et_volume': float(clean_volumes['et']),
            'removed_voxels': int(total_removed),
            'removed_in_gt': int(removed_in_gt),
            'removed_in_correct_class': int(removed_in_correct_class)
        }
        
        results.append(case_result)
        
        print(f"[{i+1}/{len(ds)}] {case_id}")
        print(f"  Raw:    WT={raw_metrics.wt:.4f} TC={raw_metrics.tc:.4f} ET={raw_metrics.et:.4f} Mean={raw_metrics.mean:.4f}")
        print(f"  Cleaned: WT={clean_metrics.wt:.4f} TC={clean_metrics.tc:.4f} ET={clean_metrics.et:.4f} Mean={clean_metrics.mean:.4f}")
        print(f"  Removed: {total_removed} voxels, {removed_in_gt} in GT, {removed_in_correct_class} in correct class")
    
    # Calculate dataset-level statistics
    raw_wt_dices = [r['raw_wt_dice'] for r in results]
    raw_tc_dices = [r['raw_tc_dice'] for r in results]
    raw_et_dices = [r['raw_et_dice'] for r in results]
    raw_mean_dices = [r['raw_mean_dice'] for r in results]
    
    clean_wt_dices = [r['clean_wt_dice'] for r in results]
    clean_tc_dices = [r['clean_tc_dice'] for r in results]
    clean_et_dices = [r['clean_et_dice'] for r in results]
    clean_mean_dices = [r['clean_mean_dice'] for r in results]
    
    dataset_stats = {
        'raw': {
            'wt_dice_mean': float(np.mean(raw_wt_dices)),
            'wt_dice_std': float(np.std(raw_wt_dices)),
            'tc_dice_mean': float(np.mean(raw_tc_dices)),
            'tc_dice_std': float(np.std(raw_tc_dices)),
            'et_dice_mean': float(np.mean(raw_et_dices)),
            'et_dice_std': float(np.std(raw_et_dices)),
            'mean_dice_mean': float(np.mean(raw_mean_dices)),
            'mean_dice_std': float(np.std(raw_mean_dices)),
        },
        'cleaned': {
            'wt_dice_mean': float(np.mean(clean_wt_dices)),
            'wt_dice_std': float(np.std(clean_wt_dices)),
            'tc_dice_mean': float(np.mean(clean_tc_dices)),
            'tc_dice_std': float(np.std(clean_tc_dices)),
            'et_dice_mean': float(np.mean(clean_et_dices)),
            'et_dice_std': float(np.std(clean_et_dices)),
            'mean_dice_mean': float(np.mean(clean_mean_dices)),
            'mean_dice_std': float(np.std(clean_mean_dices)),
        },
        'total_removed_voxels': sum(r['removed_voxels'] for r in results),
        'total_removed_in_gt': sum(r['removed_in_gt'] for r in results),
        'total_removed_in_correct_class': sum(r['removed_in_correct_class'] for r in results)
    }
    
    return results, dataset_stats


def main():
    parser = argparse.ArgumentParser(description="Validate post-processing impact on model performance")
    parser.add_argument("--data_dir", type=str, required=True, help="Validation data directory")
    parser.add_argument("--exp_name", type=str, default="exp_swinunetr_4class_et_fixed", help="Experiment name")
    parser.add_argument("--min_voxel_count", type=int, default=200, help="Minimum voxel count threshold")
    parser.add_argument("--keep_largest", action="store_true", help="Keep only largest component per class")
    parser.add_argument("--no_keep_largest", action="store_false", dest="keep_largest", default=True)
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    exp_dir = project_root / "outputs" / args.exp_name
    checkpoint_path = exp_dir / "checkpoints" / "best_mean_dice.pt"
    
    raw_preds_dir = exp_dir / "predictions" / "val_best_cases"
    cleaned_preds_dir = exp_dir / "predictions" / "val_best_cases_cleaned"
    cleaned_preds_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    results, dataset_stats = analyze_postprocessing_impact(
        data_dir=Path(args.data_dir),
        exp_dir=exp_dir,
        checkpoint_path=checkpoint_path,
        raw_preds_dir=raw_preds_dir,
        cleaned_preds_dir=cleaned_preds_dir,
        min_voxel_count=args.min_voxel_count,
        keep_largest_per_class=args.keep_largest
    )
    
    # Save per-case results
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "postprocessing_comparison.csv"
    
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'case_id', 'raw_wt_dice', 'clean_wt_dice', 'raw_tc_dice', 'clean_tc_dice',
            'raw_et_dice', 'clean_et_dice', 'raw_mean_dice', 'clean_mean_dice',
            'raw_wt_volume', 'clean_wt_volume', 'raw_tc_volume', 'clean_tc_volume',
            'raw_et_volume', 'clean_et_volume', 'removed_voxels', 'removed_in_gt', 'removed_in_correct_class'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSaved detailed comparison to: {csv_path}")
    
    # Print dataset-level statistics
    print("\n" + "=" * 60)
    print("DATASET-LEVEL STATISTICS")
    print("=" * 60)
    
    print("\nRAW PREDICTIONS:")
    print(f"  WT Dice:   {dataset_stats['raw']['wt_dice_mean']:.4f} ± {dataset_stats['raw']['wt_dice_std']:.4f}")
    print(f"  TC Dice:   {dataset_stats['raw']['tc_dice_mean']:.4f} ± {dataset_stats['raw']['tc_dice_std']:.4f}")
    print(f"  ET Dice:   {dataset_stats['raw']['et_dice_mean']:.4f} ± {dataset_stats['raw']['et_dice_std']:.4f}")
    print(f"  Mean Dice: {dataset_stats['raw']['mean_dice_mean']:.4f} ± {dataset_stats['raw']['mean_dice_std']:.4f}")
    
    print("\nCLEANED PREDICTIONS:")
    print(f"  WT Dice:   {dataset_stats['cleaned']['wt_dice_mean']:.4f} ± {dataset_stats['cleaned']['wt_dice_std']:.4f}")
    print(f"  TC Dice:   {dataset_stats['cleaned']['tc_dice_mean']:.4f} ± {dataset_stats['cleaned']['tc_dice_std']:.4f}")
    print(f"  ET Dice:   {dataset_stats['cleaned']['et_dice_mean']:.4f} ± {dataset_stats['cleaned']['et_dice_std']:.4f}")
    print(f"  Mean Dice: {dataset_stats['cleaned']['mean_dice_mean']:.4f} ± {dataset_stats['cleaned']['mean_dice_std']:.4f}")
    
    print("\nPOST-PROCESSING IMPACT:")
    print(f"  Total removed voxels: {dataset_stats['total_removed_voxels']}")
    print(f"  Removed voxels in GT: {dataset_stats['total_removed_in_gt']}")
    print(f"  Removed voxels in correct class: {dataset_stats['total_removed_in_correct_class']}")
    print(f"  Removed voxels NOT in GT: {dataset_stats['total_removed_voxels'] - dataset_stats['total_removed_in_gt']}")
    
    # Calculate improvements
    wt_change = dataset_stats['cleaned']['wt_dice_mean'] - dataset_stats['raw']['wt_dice_mean']
    tc_change = dataset_stats['cleaned']['tc_dice_mean'] - dataset_stats['raw']['tc_dice_mean']
    et_change = dataset_stats['cleaned']['et_dice_mean'] - dataset_stats['raw']['et_dice_mean']
    mean_change = dataset_stats['cleaned']['mean_dice_mean'] - dataset_stats['raw']['mean_dice_mean']
    
    print(f"\nDICE CHANGES (Cleaned - Raw):")
    print(f"  WT:   {wt_change:+.4f}")
    print(f"  TC:   {tc_change:+.4f}")
    print(f"  ET:   {et_change:+.4f}")
    print(f"  Mean: {mean_change:+.4f}")
    
    # Count cases that improved/worsened
    wt_improved = sum(1 for r in results if r['clean_wt_dice'] > r['raw_wt_dice'])
    wt_worsened = sum(1 for r in results if r['clean_wt_dice'] < r['raw_wt_dice'])
    tc_improved = sum(1 for r in results if r['clean_tc_dice'] > r['raw_tc_dice'])
    tc_worsened = sum(1 for r in results if r['clean_tc_dice'] < r['raw_tc_dice'])
    et_improved = sum(1 for r in results if r['clean_et_dice'] > r['raw_et_dice'])
    et_worsened = sum(1 for r in results if r['clean_et_dice'] < r['raw_et_dice'])
    mean_improved = sum(1 for r in results if r['clean_mean_dice'] > r['raw_mean_dice'])
    mean_worsened = sum(1 for r in results if r['clean_mean_dice'] < r['raw_mean_dice'])
    
    print(f"\nCASE-LEVEL CHANGES:")
    print(f"  WT:   {wt_improved} improved, {wt_worsened} worsened")
    print(f"  TC:   {tc_improved} improved, {tc_worsened} worsened")
    print(f"  ET:   {et_improved} improved, {et_worsened} worsened")
    print(f"  Mean: {mean_improved} improved, {mean_worsened} worsened")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if mean_change > 0.01:
        print(f"Post-processing IMPROVES mean Dice by {mean_change:.4f}")
        print(f"  {mean_improved}/{len(results)} cases improved, {mean_worsened}/{len(results)} cases worsened")
        print(f"  {dataset_stats['total_removed_voxels'] - dataset_stats['total_removed_in_gt']} removed voxels were true false positives (not in GT)")
        print(f"  {dataset_stats['total_removed_in_correct_class']} removed voxels were in correct class (potential over-filtering)")
        print("\nRECOMMENDATION: Use CLEANED predictions for paper visualization")
        print("Report post-processing as part of the complete inference pipeline.")
        print("Note: Report both raw and cleaned metrics for transparency.")
    elif mean_change < -0.01:
        print(f"Post-processing HURTS mean Dice by {mean_change:.4f}")
        print(f"  {mean_improved}/{len(results)} cases improved, {mean_worsened}/{len(results)} cases worsened")
        print(f"  {dataset_stats['total_removed_in_correct_class']} removed voxels were in correct class (significant over-filtering)")
        print("\nRECOMMENDATION: Use RAW predictions as primary scientific result")
        print("Use cleaned visualization only for qualitative comparison.")
        print("Do not report cleaned metrics as primary results.")
    else:
        print(f"Post-processing has negligible impact on mean Dice ({mean_change:.4f})")
        print(f"  {mean_improved}/{len(results)} cases improved, {mean_worsened}/{len(results)} cases worsened")
        print(f"  {dataset_stats['total_removed_voxels'] - dataset_stats['total_removed_in_gt']} removed voxels were true false positives")
        print("\nRECOMMENDATION: Use RAW predictions for paper visualization")
        print("Show RAW vs CLEANED as an ablation/post-processing experiment.")
        print("Report both versions for comprehensive analysis.")
    
    # Save dataset statistics
    stats_path = metrics_dir / "postprocessing_stats.json"
    import json
    with open(stats_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    print(f"\nDataset statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
"""Isolated segmentation evaluation utilities for prediction vs ground-truth comparison."""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

from datasets.brats_dataset import _load_nii
from utils.brats_metrics import BraTSRegionDice, _region_masks, compute_region_dice

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionMetricValues:
    dice: float | None
    hd95_mm: float | None
    sensitivity: float | None
    specificity: float | None


def _validation_print(message: str) -> None:
    """Always emit validation diagnostics to stdout for backend terminal visibility."""
    print(message, flush=True)
    logger.info(message)


def _normalize_segmentation_labels(mask: np.ndarray) -> np.ndarray:
    """Normalize BraTS/classic labels to internal {0,1,2,3}."""
    mask_i = np.rint(mask).astype(np.int16)
    unique = {int(v) for v in np.unique(mask_i) if np.isfinite(v)}
    unexpected = unique - {0, 1, 2, 3, 4}
    if unexpected:
        raise ValueError(
            f"Unexpected segmentation labels {sorted(unexpected)}. "
            "Expected BraTS labels in {{0, 1, 2, 3}} or classic BraTS {{0, 1, 2, 4}}."
        )

    out = np.zeros_like(mask_i, dtype=np.int16)
    out[mask_i == 1] = 1
    out[mask_i == 2] = 2
    out[(mask_i == 3) | (mask_i == 4)] = 3
    return out


def _spacing_for_transposed_volume(canonical_img: nib.Nifti1Image) -> tuple[float, float, float]:
    zooms = canonical_img.header.get_zooms()[:3]
    return (float(zooms[2]), float(zooms[0]), float(zooms[1]))


def _load_internal_volume(path: Path) -> tuple[nib.Nifti1Image, np.ndarray, tuple[float, float, float]]:
    """Load segmentation using the same canonical + transpose convention as inference."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    img = nib.as_closest_canonical(nib.load(str(path)))
    array = _load_nii(path)
    spacing = _spacing_for_transposed_volume(img)
    return img, array, spacing


def _prepare_aligned_label_maps(
    prediction_path: Path,
    ground_truth_path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], dict[str, Any]]:
    """Load, align, and normalize prediction and ground-truth label maps."""
    _validation_print(f"[VALIDATION] Ground truth path: {ground_truth_path}")
    _validation_print(f"[VALIDATION] Prediction path: {prediction_path}")
    _validation_print(f"[VALIDATION] Ground truth exists: {ground_truth_path.exists()}")
    _validation_print(f"[VALIDATION] Prediction exists: {prediction_path.exists()}")

    pred_img, pred_raw, pred_spacing = _load_internal_volume(prediction_path)
    gt_img, gt_raw, gt_spacing = _load_internal_volume(ground_truth_path)

    debug: dict[str, Any] = {
        "prediction_path": str(prediction_path),
        "ground_truth_path": str(ground_truth_path),
        "prediction_shape_native": tuple(int(v) for v in pred_img.shape[:3]),
        "ground_truth_shape_native": tuple(int(v) for v in gt_img.shape[:3]),
        "prediction_shape_internal": tuple(int(v) for v in pred_raw.shape),
        "ground_truth_shape_internal": tuple(int(v) for v in gt_raw.shape),
        "prediction_dtype": str(pred_raw.dtype),
        "ground_truth_dtype": str(gt_raw.dtype),
        "prediction_labels_raw": sorted(int(v) for v in np.unique(np.rint(pred_raw)).tolist() if np.isfinite(v)),
        "ground_truth_labels_raw": sorted(int(v) for v in np.unique(np.rint(gt_raw)).tolist() if np.isfinite(v)),
        "prediction_spacing": pred_spacing,
        "ground_truth_spacing": gt_spacing,
        "prediction_affine": pred_img.affine.tolist(),
        "ground_truth_affine": gt_img.affine.tolist(),
    }

    _validation_print(f"[VALIDATION] GT shape: {debug['ground_truth_shape_internal']}")
    _validation_print(f"[VALIDATION] Prediction shape: {debug['prediction_shape_internal']}")
    _validation_print(f"[VALIDATION] GT dtype: {debug['ground_truth_dtype']}")
    _validation_print(f"[VALIDATION] Prediction dtype: {debug['prediction_dtype']}")
    _validation_print(f"[VALIDATION] GT unique labels: {debug['ground_truth_labels_raw']}")
    _validation_print(f"[VALIDATION] Prediction unique labels: {debug['prediction_labels_raw']}")
    _validation_print(f"[VALIDATION] GT spacing: {debug['ground_truth_spacing']}")
    _validation_print(f"[VALIDATION] Prediction spacing: {debug['prediction_spacing']}")
    _validation_print(f"[VALIDATION] GT affine: {debug['ground_truth_affine']}")
    _validation_print(f"[VALIDATION] Prediction affine: {debug['prediction_affine']}")

    shape_match = pred_img.shape[:3] == gt_img.shape[:3]
    affine_match = np.allclose(pred_img.affine, gt_img.affine, atol=1e-3)
    debug["shape_match_native"] = shape_match
    debug["affine_match_native"] = bool(affine_match)

    if pred_raw.shape != gt_raw.shape:
        _validation_print("[VALIDATION] SHAPE MISMATCH")
        _validation_print(f"[VALIDATION] GT: {gt_raw.shape}")
        _validation_print(f"[VALIDATION] Prediction: {pred_raw.shape}")

    if not shape_match or not affine_match:
        _validation_print(
            "[VALIDATION] Resampling ground truth to prediction grid (nearest-neighbor) "
            f"shape_match={shape_match}, affine_match={affine_match}"
        )
        gt_img = resample_from_to(gt_img, pred_img, order=0)
        gt_raw = np.transpose(gt_img.get_fdata(dtype=np.float32), (2, 0, 1))
        gt_spacing = _spacing_for_transposed_volume(gt_img)
        debug["resampled_ground_truth"] = True
        debug["ground_truth_shape_resampled"] = tuple(int(v) for v in gt_raw.shape)
        _validation_print(f"[VALIDATION] GT shape after resample: {gt_raw.shape}")
    else:
        debug["resampled_ground_truth"] = False

    if pred_raw.shape != gt_raw.shape:
        raise ValueError(
            "Prediction and ground-truth volumes have incompatible spatial dimensions after "
            f"alignment: pred={pred_raw.shape}, gt={gt_raw.shape}. "
            "The uploaded ground-truth segmentation may not correspond to this MRI case."
        )

    pred_map = _normalize_segmentation_labels(pred_raw)
    gt_map = _normalize_segmentation_labels(gt_raw)

    debug["prediction_labels_normalized"] = sorted(int(v) for v in np.unique(pred_map).tolist())
    debug["ground_truth_labels_normalized"] = sorted(int(v) for v in np.unique(gt_map).tolist())

    spacing = pred_spacing
    if pred_spacing != gt_spacing:
        spacing = tuple((a + b) / 2.0 for a, b in zip(pred_spacing, gt_spacing, strict=True))

    _validation_print(f"[VALIDATION] Normalized GT labels: {debug['ground_truth_labels_normalized']}")
    _validation_print(f"[VALIDATION] Normalized prediction labels: {debug['prediction_labels_normalized']}")
    _validation_print(f"[VALIDATION] Using spacing (mm): {spacing}")

    return pred_map, gt_map, spacing, debug


def _binary_surface(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = binary_erosion(mask)
    return mask & ~eroded


def _compute_hd95_mm(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing: tuple[float, float, float]) -> float | None:
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return None

    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    if not pred_bin.any() and not gt_bin.any():
        return 0.0
    if not pred_bin.any() or not gt_bin.any():
        return None

    pred_surface = _binary_surface(pred_bin)
    gt_surface = _binary_surface(gt_bin)
    if not pred_surface.any() or not gt_surface.any():
        return None

    sp = np.asarray(spacing, dtype=float)
    dt_gt = distance_transform_edt(~gt_bin, sampling=sp)
    dt_pred = distance_transform_edt(~pred_bin, sampling=sp)

    dist_pred_to_gt = dt_gt[pred_surface]
    dist_gt_to_pred = dt_pred[gt_surface]
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    if all_distances.size == 0:
        return None
    return float(np.percentile(all_distances, 95))


def _compute_sensitivity_specificity(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple[float, float]:
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin, gt_bin).sum())
    tn = int(np.logical_and(~pred_bin, ~gt_bin).sum())

    sensitivity = 1.0 if tp + fn == 0 else float(tp / (tp + fn))
    specificity = 1.0 if tn + fp == 0 else float(tn / (tn + fp))
    return sensitivity, specificity


def _region_binary_masks(pred_map: np.ndarray, gt_map: np.ndarray, region: str) -> tuple[np.ndarray, np.ndarray]:
    pred_t = torch.from_numpy(pred_map).unsqueeze(0)
    gt_t = torch.from_numpy(gt_map).unsqueeze(0)
    pred_bin = _region_masks(pred_t, region).squeeze(0).numpy()
    gt_bin = _region_masks(gt_t, region).squeeze(0).numpy()
    return pred_bin, gt_bin


def _compute_region_metrics_individually(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    region: str,
    spacing: tuple[float, float, float],
    dice_value: float,
) -> tuple[RegionMetricValues, dict[str, str]]:
    warnings: dict[str, str] = {}
    region_upper = region.upper()
    pred_bin, gt_bin = _region_binary_masks(pred_map, gt_map, region)

    dice = round(float(dice_value), 4)
    _validation_print(f"[VALIDATION] {region_upper} Dice OK: {dice}")

    hd95: float | None
    try:
        hd95_raw = _compute_hd95_mm(pred_bin, gt_bin, spacing)
        hd95 = round(float(hd95_raw), 4) if hd95_raw is not None else None
        _validation_print(f"[VALIDATION] {region_upper} HD95 OK: {hd95}")
    except Exception as exc:
        hd95 = None
        warnings["hd95"] = str(exc)
        _validation_print(f"[VALIDATION] {region_upper} HD95 FAILED: {exc!r}")

    try:
        sensitivity, specificity = _compute_sensitivity_specificity(pred_bin, gt_bin)
        sensitivity = round(float(sensitivity), 4)
        specificity = round(float(specificity), 4)
        _validation_print(f"[VALIDATION] {region_upper} Sensitivity OK: {sensitivity}")
        _validation_print(f"[VALIDATION] {region_upper} Specificity OK: {specificity}")
    except Exception as exc:
        sensitivity = None
        specificity = None
        warnings["sensitivity"] = str(exc)
        warnings["specificity"] = str(exc)
        _validation_print(f"[VALIDATION] {region_upper} Sensitivity/Specificity FAILED: {exc!r}")

    return (
        RegionMetricValues(
            dice=dice,
            hd95_mm=hd95,
            sensitivity=sensitivity,
            specificity=specificity,
        ),
        warnings,
    )


def _format_metrics_dict(
    wt: RegionMetricValues,
    tc: RegionMetricValues,
    et: RegionMetricValues,
    warnings: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    def _region_payload(region: RegionMetricValues) -> dict[str, float | None]:
        return {
            "dice": region.dice,
            "hd95": region.hd95_mm,
            "hd95_mm": region.hd95_mm,
            "sensitivity": region.sensitivity,
            "specificity": region.specificity,
        }

    payload: dict[str, Any] = {
        "available": True,
        "ground_truth_available": True,
        "metrics": {
            "WT": _region_payload(wt),
            "TC": _region_payload(tc),
            "ET": _region_payload(et),
        },
        "wt": asdict(wt),
        "tc": asdict(tc),
        "et": asdict(et),
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


def evaluate_segmentation_masks(
    prediction_path: Path,
    ground_truth_path: Path,
) -> dict[str, Any]:
    """Compute WT/TC/ET metrics; returns structured success or failure payload."""
    try:
        pred_map, gt_map, spacing, _ = _prepare_aligned_label_maps(prediction_path, ground_truth_path)

        pred_tensor = torch.from_numpy(pred_map).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_map).unsqueeze(0)
        dice_scores: BraTSRegionDice = compute_region_dice(pred_tensor, gt_tensor, from_logits=False)

        all_warnings: dict[str, dict[str, str]] = {}
        wt, wt_warn = _compute_region_metrics_individually(pred_map, gt_map, "wt", spacing, dice_scores.wt)
        tc, tc_warn = _compute_region_metrics_individually(pred_map, gt_map, "tc", spacing, dice_scores.tc)
        et, et_warn = _compute_region_metrics_individually(pred_map, gt_map, "et", spacing, dice_scores.et)

        if wt_warn:
            all_warnings["WT"] = wt_warn
        if tc_warn:
            all_warnings["TC"] = tc_warn
        if et_warn:
            all_warnings["ET"] = et_warn

        result = _format_metrics_dict(wt, tc, et, all_warnings or None)
        _validation_print(f"[VALIDATION] SUCCESS: {result['metrics']}")
        return result
    except Exception as exc:
        _validation_print(f"[VALIDATION] ERROR: {exc!r}")
        traceback.print_exc()
        return {
            "available": False,
            "ground_truth_available": True,
            "reason": str(exc),
            "error_type": type(exc).__name__,
            "metrics": None,
        }


def generate_comparison_mask(
    prediction_path: Path,
    ground_truth_path: Path,
    output_path: Path,
    reference_nifti_path: Path,
) -> Path:
    """Generate a difference label map: 1=TP, 2=FP, 3=FN (WT region)."""
    pred_map, gt_map, _, _ = _prepare_aligned_label_maps(prediction_path, ground_truth_path)

    pred_wt = (pred_map == 1) | (pred_map == 2) | (pred_map == 3)
    gt_wt = (gt_map == 1) | (gt_map == 2) | (gt_map == 3)

    comparison = np.zeros_like(pred_map, dtype=np.uint8)
    comparison[np.logical_and(pred_wt, gt_wt)] = 1
    comparison[np.logical_and(pred_wt, ~gt_wt)] = 2
    comparison[np.logical_and(~pred_wt, gt_wt)] = 3

    ref_loaded = nib.as_closest_canonical(nib.load(str(reference_nifti_path)))
    comparison_transposed = np.transpose(comparison, (1, 2, 0))
    out_img = nib.Nifti1Image(comparison_transposed.astype(np.uint8), ref_loaded.affine, ref_loaded.header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))
    return output_path

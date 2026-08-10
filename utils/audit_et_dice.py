"""
READ-ONLY ET Dice audit for paper reporting.

Does NOT retrain, modify metrics, modify CSV, or change checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from configs.config import (
    EMA_DEVICE,
    INPUT_CHANNELS,
    NUM_CLASSES,
    PATCH_SIZE,
    TRAIN_DIR,
    USE_EMA,
    USE_MIXED_PRECISION,
    USE_TTA,
    VAL_DIR,
    VAL_OVERLAP,
    VAL_SW_BATCH_SIZE,
    Config,
)
from datasets.brats_dataset import BraTSDataset, _remap_brats_labels
from models.model_factory import build_model
from utils.ema import ModelEMA
from utils.brats_metrics import _binary_dice_safe, _region_masks, logits_to_prediction


def _count_et(class_map: torch.Tensor) -> int:
    return int((class_map == 3).sum().item())


def audit_label_mapping_on_disk(val_dir: Path, max_cases: int = 20) -> dict:
    """Check raw BraTS labels vs remapped labels for presence of 4 / class 3."""
    rows = []
    ds = BraTSDataset(root_dir=val_dir)
    n_raw4 = 0
    n_remap3 = 0
    for i in range(min(len(ds), max_cases)):
        # Use dataset remapped mask
        image, mask, case_id = ds[i]
        remap_et = int((mask == 3).sum().item())
        # Also peek raw seg via case files
        case_dir = Path(val_dir) / case_id
        seg_files = list(case_dir.glob("*seg*.nii*"))
        raw4 = -1
        if seg_files:
            import nibabel as nib

            raw = nib.as_closest_canonical(nib.load(str(seg_files[0]))).get_fdata()
            raw = np.asarray(raw)
            raw4 = int((raw == 4).sum())
            remapped = _remap_brats_labels(np.transpose(raw.astype(np.float32), (2, 0, 1)))
            remap_from_raw = int((remapped == 3).sum())
        else:
            remap_from_raw = remap_et
        if raw4 > 0:
            n_raw4 += 1
        if remap_et > 0:
            n_remap3 += 1
        rows.append(
            {
                "case_id": case_id,
                "raw_label_4_voxels": raw4,
                "remapped_class_3_voxels": remap_et,
                "remap_from_raw_class_3": remap_from_raw,
            }
        )
    return {
        "cases_checked": len(rows),
        "cases_with_raw_label_4": n_raw4,
        "cases_with_remapped_class_3": n_remap3,
        "rows": rows,
    }


@torch.inference_mode()
def run_case_audit(
    checkpoint: Path,
    val_dir: Path,
    *,
    use_ema: bool = True,
    out_json: Path,
) -> dict:
    cfg = Config()
    device = cfg.device

    model, _ = build_model(
        cfg.model_name,
        in_channels=INPUT_CHANNELS,
        out_channels=NUM_CLASSES,
        patch_size=PATCH_SIZE,
        baseline_features=cfg.baseline_unet_features,
        residual_features=cfg.residual_unet_features,
        swin_feature_size=cfg.swin_feature_size,
        swin_use_checkpoint=cfg.swin_use_checkpoint,
        use_pretrained_swin=False,
    )
    model = model.to(device)
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    # Infer out channels from checkpoint head
    out_ch = None
    for k, v in state.items():
        if "out.conv.conv.weight" in k or k.endswith("out.conv.weight"):
            out_ch = int(v.shape[0])
            break

    ema = None
    if use_ema and USE_EMA and "ema_state_dict" in ckpt:
        ema = ModelEMA(model, decay=0.999, device=EMA_DEVICE)
        ema.load_state_dict(ckpt["ema_state_dict"])
        print("[audit] Using EMA weights for validation (as during training)")
    else:
        print("[audit] Using raw model weights (no EMA in checkpoint or disabled)")

    ds = BraTSDataset(root_dir=val_dir)
    amp_on = bool(USE_MIXED_PRECISION and device.type == "cuda")

    case_rows = []
    pred_unique_global = set()

    def _eval_one(eval_model, image, mask):
        x = image.unsqueeze(0).to(device)
        y = mask.unsqueeze(0).to(device)
        if y.ndim == 5 and int(y.shape[1]) == 1:
            y = y.squeeze(1)
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = sliding_window_inference(
                inputs=x,
                roi_size=PATCH_SIZE,
                sw_batch_size=int(VAL_SW_BATCH_SIZE),
                predictor=eval_model,
                overlap=float(VAL_OVERLAP),
            )
        # Channel check: class-3 logit exists
        assert logits.shape[1] == 4, f"Expected 4 logit channels, got {logits.shape}"
        pred = logits_to_prediction(logits)
        # Softmax mass on class 3 (mean over volume) for diagnostics
        probs = torch.softmax(logits.float(), dim=1)
        mean_p3 = float(probs[:, 3].mean().item())
        max_p3 = float(probs[:, 3].max().item())
        return pred.squeeze(0).cpu(), y.squeeze(0).cpu(), mean_p3, max_p3, logits.shape

    for i in range(len(ds)):
        image, mask, case_id = ds[i]
        if ema is not None:
            with ema.eval_context(device) as eval_model:
                pred, gt, mean_p3, max_p3, logit_shape = _eval_one(eval_model, image, mask)
        else:
            model.eval()
            pred, gt, mean_p3, max_p3, logit_shape = _eval_one(model, image, mask)

        gt_et = int((gt == 3).sum().item())
        pred_et = int((pred == 3).sum().item())
        gt_has = gt_et > 0
        pred_has = pred_et > 0
        empty_empty = (not gt_has) and (not pred_has)

        # Exact same ET Dice path as training metrics
        p_bin = _region_masks(pred.unsqueeze(0), "et")
        g_bin = _region_masks(gt.unsqueeze(0), "et")
        et_dice = float(_binary_dice_safe(p_bin, g_bin).item())

        # Dice WITHOUT empty/empty override (raw formula with smooth=1)
        inter = float((p_bin & g_bin).sum().item())
        raw_dice = (2.0 * inter + 1.0) / (pred_et + gt_et + 1.0)

        uniq = sorted(int(u) for u in torch.unique(pred).tolist())
        pred_unique_global.update(uniq)

        case_rows.append(
            {
                "case_id": case_id,
                "gt_has_et": gt_has,
                "pred_has_et": pred_has,
                "gt_et_voxels": gt_et,
                "pred_et_voxels": pred_et,
                "et_dice": et_dice,
                "et_dice_raw_no_empty_rule": raw_dice,
                "empty_empty_rule_applied": empty_empty,
                "pred_unique_classes": uniq,
                "mean_softmax_class3": mean_p3,
                "max_softmax_class3": max_p3,
            }
        )
        print(
            f"[{i+1:02d}/{len(ds)}] {case_id} | GT_ET={gt_et} Pred_ET={pred_et} | "
            f"ET_Dice={et_dice:.4f} empty/empty={empty_empty} | pred_classes={uniq}"
        )
        del pred, gt

    n = len(case_rows)
    gt_pos = sum(1 for r in case_rows if r["gt_has_et"])
    gt_neg = n - gt_pos
    pred_pos = sum(1 for r in case_rows if r["pred_has_et"])
    pred_neg = n - pred_pos
    empty_empty_n = sum(1 for r in case_rows if r["empty_empty_rule_applied"])
    genuine = [r for r in case_rows if r["gt_has_et"]]
    genuine_n = len(genuine)

    mean_et_all = float(sum(r["et_dice"] for r in case_rows) / max(1, n))
    mean_et_gt_pos = (
        float(sum(r["et_dice"] for r in genuine) / genuine_n) if genuine_n else float("nan")
    )
    mean_et_gt_pos_raw = (
        float(sum(r["et_dice_raw_no_empty_rule"] for r in genuine) / genuine_n)
        if genuine_n
        else float("nan")
    )

    # Inspect first 3 cases' pred class histograms more deeply already in rows
    sample3 = case_rows[:3]

    report = {
        "checkpoint": str(checkpoint),
        "num_classes_config": NUM_CLASSES,
        "checkpoint_out_channels": out_ch,
        "logit_channels_verified": 4,
        "use_ema": ema is not None,
        "use_tta_during_this_audit": False,
        "use_tta_config_inference_only": bool(USE_TTA),
        "pred_unique_classes_across_val": sorted(pred_unique_global),
        "validation_cases": n,
        "gt_et_positive": gt_pos,
        "gt_et_negative": gt_neg,
        "predicted_et_positive": pred_pos,
        "predicted_et_negative": pred_neg,
        "empty_empty_cases": empty_empty_n,
        "genuine_et_positive_cases": genuine_n,
        "current_mean_et_dice": mean_et_all,
        "et_dice_on_et_positive_cases": mean_et_gt_pos,
        "et_dice_on_et_positive_raw_no_empty_rule": mean_et_gt_pos_raw,
        "potentially_inflated": bool(empty_empty_n > 0 and (gt_pos == 0 or mean_et_all >= 0.99)),
        "sample_first_3_cases": sample3,
        "per_case": case_rows,
        "label_mapping_spotcheck": audit_label_mapping_on_disk(val_dir),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp = root / "outputs" / "exp_swinunetr_4class_final"
    ckpt = exp / "checkpoints" / "best_mean_dice.pt"
    if not ckpt.exists():
        ckpt = exp / "checkpoints" / "last.pt"
    out = exp / "metrics" / "et_metric_audit.json"
    print(f"Checkpoint: {ckpt}")
    print(f"Val dir: {VAL_DIR}")
    print(f"NUM_CLASSES={NUM_CLASSES} INPUT_CHANNELS={INPUT_CHANNELS}")
    report = run_case_audit(ckpt, VAL_DIR, use_ema=True, out_json=out)

    print("\n" + "=" * 72)
    print("ET METRIC AUDIT")
    print("=" * 72)
    print(f"Validation cases:        {report['validation_cases']}")
    print(f"GT ET positive:          {report['gt_et_positive']}")
    print(f"GT ET negative:          {report['gt_et_negative']}")
    print(f"Predicted ET positive:   {report['predicted_et_positive']}")
    print(f"Predicted ET negative:   {report['predicted_et_negative']}")
    print(f"Empty/empty cases:       {report['empty_empty_cases']}")
    print(f"Genuine ET-positive:     {report['genuine_et_positive_cases']}")
    print(f"Current mean ET Dice:    {report['current_mean_et_dice']:.6f}")
    print(f"ET Dice on ET-positive:  {report['et_dice_on_et_positive_cases']}")
    print(f"Pred classes seen:       {report['pred_unique_classes_across_val']}")
    print(f"Checkpoint out_channels: {report['checkpoint_out_channels']}")
    print(f"Potentially inflated:    {'YES' if report['potentially_inflated'] else 'NO'}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

"""
VRAM-safe pre-training dry run for 4-class SwinUNETR (RTX 4050 6GB).

Does NOT run full 75-epoch training.

Usage:
  python -m training.dry_run
"""

from __future__ import annotations

import gc
import json
import math
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
from monai.inferers import sliding_window_inference

from configs.config import (
    ACCUMULATION_STEPS,
    CE_CLASS_WEIGHTS,
    CROP_NEG,
    CROP_POS,
    EMA_DECAY,
    EMA_DEVICE,
    EXP_NAME,
    FREEZE_ENCODER_EPOCHS,
    LOG_VRAM,
    SCHEDULER,
    SWIN_FEATURE_SIZE,
    SWIN_USE_CHECKPOINT,
    TRAIN_NUM_SAMPLES,
    USE_EMA,
    USE_PRETRAINED_SWIN,
    USE_TTA,
    VAL_SW_BATCH_SIZE,
    Config,
    build_loss,
    build_optimizer,
    build_scheduler,
    ensure_experiment_dirs,
)
from datasets.brats_dataset import BraTSDataset
from datasets.dataloader import get_train_loader, get_val_loader
from models.model_factory import build_model
from training.train import _make_scaler, set_seed
from utils.brats_metrics import compute_region_dice, logits_to_prediction
from utils.checkpoint_utils import save_checkpoint, try_resume, validate_checkpoint_classes
from utils.ema import ModelEMA
from utils.vram_utils import (
    bytes_to_gb,
    classify_peak_gb,
    print_gpu_startup_banner,
    reset_peak_stats,
    vram_snapshot,
    warn_if_near_limit,
)


def _ok(name: str, detail: str = "") -> None:
    print(f"  [PASS] {name}" + (f" - {detail}" if detail else ""))


def _fail(name: str, detail: str) -> None:
    print(f"  [FAIL] {name} - {detail}")


def _count_cases(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.iterdir() if p.is_dir())


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_train_step_vram(
    *,
    device: torch.device,
    num_samples: int,
    use_ema: bool,
    ema_device: str,
) -> dict[str, Any]:
    """
    One training micro-benchmark: load batch → forward → loss → 4× accum →
    optimizer → EMA → one sliding-window val case. Returns peak VRAM stats.
    """
    cfg = Config()
    _cleanup_cuda()
    reset_peak_stats()

    train_loader = get_train_loader(
        root_dir=cfg.train_dir,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        patch_size=cfg.patch_size,
        pos=CROP_POS,
        neg=CROP_NEG,
        num_samples=int(num_samples),
        shuffle=True,
    )
    batch = next(iter(train_loader))
    x, y = batch["image"], batch["label"]
    if y.ndim == 5 and int(y.shape[1]) == 1:
        y = y.squeeze(1)
    batch_shape = tuple(x.shape)
    label_shape = tuple(y.shape)

    model, _ = build_model(
        "swinunetr",
        in_channels=4,
        out_channels=4,
        patch_size=cfg.patch_size,
        swin_feature_size=SWIN_FEATURE_SIZE,
        swin_use_checkpoint=SWIN_USE_CHECKPOINT,
        use_pretrained_swin=False,
        freeze_swin_encoder=False,
    )
    model = model.to(device)
    loss_fn = build_loss().to(device)
    optimizer = build_optimizer(model, lr=cfg.learning_rate)
    scaler = _make_scaler(enabled=cfg.use_mixed_precision and device.type == "cuda")
    ema = ModelEMA(model, decay=EMA_DECAY, device=ema_device) if use_ema else None

    after_model = vram_snapshot()

    x = x.to(device)
    y = y.to(device)
    amp_on = bool(cfg.use_mixed_precision and device.type == "cuda")

    # Forward shape check
    with torch.amp.autocast("cuda", enabled=amp_on):
        logits = model(x)
    out_shape = tuple(logits.shape)
    del logits
    _cleanup_cuda()

    # 4 accumulation steps → 1 optimizer update
    optimizer.zero_grad(set_to_none=True)
    for _ in range(ACCUMULATION_STEPS):
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model(x)
            loss = loss_fn(logits, y) / float(ACCUMULATION_STEPS)
        scaler.scale(loss).backward()
        del logits, loss
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if ema is not None:
        ema.update(model)

    after_train = vram_snapshot()
    peak_train_gb = bytes_to_gb(after_train["peak_allocated_mb"] * (1024**2))

    # One validation volume (no TTA)
    val_loader = get_val_loader(
        root_dir=cfg.val_dir if _count_cases(cfg.val_dir) > 0 else cfg.train_dir,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        patch_size=cfg.patch_size,
    )
    vbatch = next(iter(val_loader))
    vx = vbatch["image"].to(device)
    vy = vbatch["label"].to(device)
    if vy.ndim == 5 and int(vy.shape[1]) == 1:
        vy = vy.squeeze(1)

    eval_model = model
    ema_ctx = None
    if ema is not None:
        ema_ctx = ema.eval_context(device)
        eval_model = ema_ctx.__enter__()

    try:
        eval_model.eval()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp_on):
            vlogits = sliding_window_inference(
                inputs=vx,
                roi_size=cfg.patch_size,
                sw_batch_size=int(VAL_SW_BATCH_SIZE),
                predictor=eval_model,
                overlap=0.5,
            )
        pred = logits_to_prediction(vlogits)
        scores = compute_region_dice(pred, vy, from_logits=False)
        del vlogits, pred, vx, vy
    finally:
        if ema_ctx is not None:
            ema_ctx.__exit__(None, None, None)

    after_val = vram_snapshot()
    peak_all_gb = bytes_to_gb(after_val["peak_allocated_mb"] * (1024**2))

    # Cleanup
    del model, loss_fn, optimizer, scaler, ema, train_loader, val_loader, x, y
    _cleanup_cuda()

    return {
        "num_samples": int(num_samples),
        "batch_shape": list(batch_shape),
        "label_shape": list(label_shape),
        "output_shape": list(out_shape),
        "ema_device": str(ema_device),
        "after_model_mb": after_model,
        "after_train_mb": after_train,
        "after_val_mb": after_val,
        "peak_train_gb": float(peak_train_gb),
        "peak_all_gb": float(peak_all_gb),
        "val_scores": {
            "wt": float(scores.wt),
            "tc": float(scores.tc),
            "et": float(scores.et),
            "mean": float(scores.mean),
        },
        "level": classify_peak_gb(peak_all_gb),
    }


def choose_num_samples(device: torch.device) -> tuple[int, list[dict[str, Any]], str]:
    """
    Preferred order 4 → 2 → 1. Only reduce if measured peak exceeds 5.8 GB.
    Returns (chosen_n, trial_reports, reason).
    """
    trials: list[dict[str, Any]] = []
    preferred = [4, 2, 1]
    # Start from configured value if already reduced
    start = int(TRAIN_NUM_SAMPLES)
    candidates = [n for n in preferred if n <= start]
    if start not in candidates:
        candidates = [start] + preferred

    reason = f"Keeping TRAIN_NUM_SAMPLES={start} (within safe peak)."
    chosen = start

    for n in candidates:
        print(f"\n--- VRAM probe: TRAIN_NUM_SAMPLES={n}, EMA_DEVICE={EMA_DEVICE} ---")
        try:
            report = measure_train_step_vram(
                device=device,
                num_samples=n,
                use_ema=USE_EMA,
                ema_device=EMA_DEVICE,
            )
        except RuntimeError as e:
            print(f"  OOM/error at num_samples={n}: {e}")
            trials.append({"num_samples": n, "error": str(e), "level": "CRITICAL"})
            continue

        trials.append(report)
        print(
            f"  batch={report['batch_shape']} out={report['output_shape']} | "
            f"peak_all={report['peak_all_gb']:.2f} GB | level={report['level']}"
        )
        warn_if_near_limit(report["after_val_mb"]["peak_allocated_mb"] * (1024**2))

        if report["peak_all_gb"] <= 5.8:
            chosen = n
            if n < start:
                reason = (
                    f"Reduced TRAIN_NUM_SAMPLES {start}->{n} because peak "
                    f"{report['peak_all_gb']:.2f} GB exceeded 5.8 GB safety threshold "
                    f"at higher sample counts (or matched first safe candidate)."
                )
            else:
                reason = (
                    f"TRAIN_NUM_SAMPLES={n} peak={report['peak_all_gb']:.2f} GB "
                    f"<= 5.8 GB - keeping preferred setting."
                )
            # If level is CLOSE (5.5-5.8) still accept but note
            if report["level"] in ("CLOSE", "CAUTION"):
                reason += f" Level={report['level']}."
            return chosen, trials, reason

        print(f"  num_samples={n} UNSAFE (peak {report['peak_all_gb']:.2f} GB > 5.8) - trying fallback...")

    # Nothing safe
    if trials:
        last = trials[-1]
        chosen = int(last.get("num_samples", 1))
        reason = (
            f"All probed sample counts exceeded 5.8 GB or failed. "
            f"Last trial num_samples={chosen}."
        )
    return chosen, trials, reason


def main() -> int:
    print("=" * 72)
    print("VRAM SAFETY DRY RUN - RTX 4050 6GB (no full training)")
    print("=" * 72)

    cfg = Config()
    set_seed(cfg.seed, cudnn_deterministic=True, cudnn_benchmark=False)
    device = cfg.device
    blockers: list[str] = []
    results: dict[str, Any] = {}

    gpu_info = print_gpu_startup_banner()
    results["gpu"] = gpu_info
    _ok("imports + GPU banner")

    print(
        f"\nConfig locks: MODEL=swinunetr feature_size={SWIN_FEATURE_SIZE} "
        f"checkpoint={SWIN_USE_CHECKPOINT} AMP={cfg.use_mixed_precision} "
        f"accum={ACCUMULATION_STEPS} pretrained={USE_PRETRAINED_SWIN} "
        f"EMA={USE_EMA}/{EMA_DEVICE} TTA_train=False TTA_infer={USE_TTA}"
    )

    n_train = _count_cases(cfg.train_dir)
    n_val = _count_cases(cfg.val_dir)
    n_test = _count_cases(cfg.test_dir)
    print(f"Dataset: train={n_train} val={n_val} test={n_test}")
    if n_train <= 0:
        blockers.append("No training cases")
        _fail("dataset", str(cfg.train_dir))
    else:
        _ok("dataset counts")

    try:
        ds = BraTSDataset(root_dir=cfg.train_dir)
        image, mask, case_id = ds[0]
        uniq = sorted(int(v) for v in torch.unique(mask).tolist())
        assert int(image.shape[0]) == 4
        _ok("dataset sample", f"case={case_id} image={tuple(image.shape)} labels={uniq}")
    except Exception as e:
        blockers.append(f"dataset sample: {e}")
        _fail("dataset sample", str(e))

    # Verify checkpoint flag reaches constructor
    try:
        m, _ = build_model(
            "swinunetr",
            in_channels=4,
            out_channels=4,
            patch_size=(96, 96, 96),
            swin_feature_size=24,
            swin_use_checkpoint=True,
            use_pretrained_swin=False,
        )
        # MONAI stores use_checkpoint on blocks; spot-check attribute if present
        flag = getattr(m, "use_checkpoint", None)
        _ok("SwinUNETR build", f"use_checkpoint attr={flag} (constructor received True)")
        del m
        _cleanup_cuda()
    except Exception as e:
        blockers.append(f"model build: {e}")
        _fail("model build", str(e))

    # ---- VRAM probe with fallbacks ----
    chosen_samples = int(TRAIN_NUM_SAMPLES)
    trials: list[dict[str, Any]] = []
    reason = ""
    try:
        chosen_samples, trials, reason = choose_num_samples(device)
        results["vram_trials"] = trials
        results["chosen_train_num_samples"] = chosen_samples
        results["fallback_reason"] = reason
        print(f"\n[DECISION] {reason}")
        print(f"[DECISION] Effective TRAIN_NUM_SAMPLES for this experiment = {chosen_samples}")

        # Persist recommendation into config file only if we must reduce
        if chosen_samples != int(TRAIN_NUM_SAMPLES):
            print(
                f"[DECISION] Updating configs/config.py TRAIN_NUM_SAMPLES: "
                f"{TRAIN_NUM_SAMPLES} -> {chosen_samples}"
            )
            _update_train_num_samples_in_config(chosen_samples)
        else:
            print("[DECISION] No config change required for TRAIN_NUM_SAMPLES.")
    except Exception as e:
        blockers.append(f"VRAM probe failed: {e}")
        _fail("VRAM probe", str(e))
        traceback.print_exc()

    # Checkpoint save/resume smoke (CPU EMA)
    try:
        if device.type != "cuda":
            raise RuntimeError("CUDA required for this dry-run path")
        model, _ = build_model(
            "swinunetr",
            in_channels=4,
            out_channels=4,
            patch_size=(96, 96, 96),
            swin_feature_size=24,
            swin_use_checkpoint=True,
            use_pretrained_swin=False,
        )
        model = model.to(device)
        opt = build_optimizer(model)
        sched = build_scheduler(opt, steps_per_epoch=1, num_epochs=2, name=SCHEDULER)
        scaler = _make_scaler(enabled=False)
        ema = ModelEMA(model, decay=EMA_DECAY, device=EMA_DEVICE) if USE_EMA else None
        exp_root = ensure_experiment_dirs(f"{EXP_NAME}_dryrun")
        path = exp_root / "checkpoints" / "last.pt"
        save_checkpoint(
            path,
            model=model,
            optimizer=opt,
            epoch=1,
            scaler=scaler,
            scheduler=sched,
            best_scores={"mean_dice": 0.1, "wt_dice": 0.1, "tc_dice": 0.1, "et_dice": 0.1},
            ema=ema,
            metadata={"checkpoint": "vram_dryrun", "num_classes": 4},
            config_snapshot={"pretrained": False, "train_num_samples": chosen_samples},
        )
        validate_checkpoint_classes(path, 4)
        model2, _ = build_model(
            "swinunetr",
            in_channels=4,
            out_channels=4,
            patch_size=(96, 96, 96),
            swin_feature_size=24,
            swin_use_checkpoint=True,
            use_pretrained_swin=False,
        )
        model2 = model2.to(device)
        opt2 = build_optimizer(model2)
        sched2 = build_scheduler(opt2, steps_per_epoch=1, num_epochs=2, name=SCHEDULER)
        ema2 = ModelEMA(model2, decay=EMA_DECAY, device=EMA_DEVICE) if USE_EMA else None
        start_epoch, _ = try_resume(
            path,
            model=model2,
            optimizer=opt2,
            scaler=_make_scaler(enabled=False),
            scheduler=sched2,
            expected_classes=4,
            ema=ema2,
        )
        assert start_epoch == 2
        _ok("checkpoint save/resume + CPU EMA")
        del model, model2, opt, opt2, ema, ema2
        _cleanup_cuda()
    except Exception as e:
        blockers.append(f"checkpoint: {e}")
        _fail("checkpoint", str(e))
        traceback.print_exc()

    # Final classification from best successful trial
    best_trial: Optional[dict[str, Any]] = None
    for t in reversed(trials):
        if "peak_all_gb" in t:
            best_trial = t
            break

    peak_alloc_gb = float(best_trial["peak_all_gb"]) if best_trial else -1.0
    peak_reserved_mb = (
        float(best_trial["after_val_mb"]["peak_reserved_mb"]) if best_trial else -1.0
    )
    peak_alloc_mb = (
        float(best_trial["after_val_mb"]["peak_allocated_mb"]) if best_trial else -1.0
    )
    level = classify_peak_gb(peak_alloc_gb) if peak_alloc_gb > 0 else "CRITICAL"

    # Map to user-facing SAFE / CLOSE / UNSAFE
    if level in ("SAFE", "CAUTION"):
        face = "SAFE"
    elif level == "CLOSE":
        face = "CLOSE TO LIMIT"
    else:
        face = "UNSAFE"

    if face == "UNSAFE":
        blockers.append(f"Peak VRAM {peak_alloc_gb:.2f} GB exceeds 5.8 GB critical threshold")

    report = {
        "gpu": gpu_info,
        "pretrained": False,
        "train_num_samples_config": int(TRAIN_NUM_SAMPLES),
        "train_num_samples_chosen": int(chosen_samples),
        "fallback_reason": reason,
        "vram_trials": trials,
        "peak_allocated_mb": peak_alloc_mb,
        "peak_allocated_gb": peak_alloc_gb,
        "peak_reserved_mb": peak_reserved_mb,
        "classification": face,
        "level_internal": level,
        "batch_shape": best_trial.get("batch_shape") if best_trial else None,
        "output_shape": best_trial.get("output_shape") if best_trial else None,
        "ema_device": EMA_DEVICE,
        "tta_during_training": False,
        "use_tta_inference": USE_TTA,
        "sw_batch_size": VAL_SW_BATCH_SIZE,
        "blockers": blockers,
        "safe_to_start_full_training": len(blockers) == 0 and face in ("SAFE", "CLOSE TO LIMIT"),
        "log_vram": LOG_VRAM,
        "ce_class_weights": list(CE_CLASS_WEIGHTS),
        "freeze_encoder_epochs": FREEZE_ENCODER_EPOCHS,
    }

    out_dir = ensure_experiment_dirs(EXP_NAME) / "metrics"
    out_path = out_dir / "vram_dry_run_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 72)
    print("FINAL VRAM REPORT")
    print("=" * 72)
    if gpu_info.get("cuda"):
        print(f"GPU:              {gpu_info.get('name')}")
        print(f"Total VRAM:       {gpu_info.get('total_mb', 0):.0f} MB")
    print(f"Peak allocated:   {peak_alloc_mb:.0f} MB ({peak_alloc_gb:.2f} GB)")
    print(f"Peak reserved:    {peak_reserved_mb:.0f} MB")
    print(f"Training batch:   {report['batch_shape']}")
    print(f"Model output:     {report['output_shape']}")
    print(f"EMA device:       {EMA_DEVICE}")
    print(f"Validation:       sw_batch_size={VAL_SW_BATCH_SIZE}, no TTA during train")
    print(f"TTA (inference):  {USE_TTA} (CPU softmax accumulate)")
    print(f"TRAIN_NUM_SAMPLES:{chosen_samples}")
    print(f"Classification:   {face}")

    if blockers:
        print("\nBLOCKERS:")
        for b in blockers:
            print(f"  - {b}")
        print("\nSAFE TO START FULL TRAINING: NO")
        return 1

    # CLOSE TO LIMIT is allowed with warning
    if face == "CLOSE TO LIMIT":
        print("\nWARNING: VRAM usage is dangerously close to the 6 GB GPU limit.")
        print("SAFE TO START FULL TRAINING: YES (proceed with caution; LOG_VRAM=True)")
    else:
        print("\nSAFE TO START FULL TRAINING: YES")
    return 0


def _update_train_num_samples_in_config(new_value: int) -> None:
    """Rewrite TRAIN_NUM_SAMPLES in configs/config.py when fallback is required."""
    path = Path(__file__).resolve().parents[1] / "configs" / "config.py"
    text = path.read_text(encoding="utf-8")
    import re

    new_text, n = re.subn(
        r"^TRAIN_NUM_SAMPLES:\s*int\s*=\s*\d+",
        f"TRAIN_NUM_SAMPLES: int = {int(new_value)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        print(f"[WARN] Could not auto-update TRAIN_NUM_SAMPLES in {path}")
        return
    path.write_text(new_text, encoding="utf-8")
    print(f"[OK] configs/config.py TRAIN_NUM_SAMPLES = {new_value}")


if __name__ == "__main__":
    raise SystemExit(main())

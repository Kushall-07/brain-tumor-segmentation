"""Training entrypoint for 4-class SwinUNETR (BraTS WT/TC/ET) — 6GB VRAM-safe."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from configs.config import (
    ACCUMULATION_STEPS,
    CE_CLASS_WEIGHTS,
    CROP_NEG,
    CROP_POS,
    EMA_DEVICE,
    EXP_NAME,
    FOCAL_GAMMA,
    FREEZE_ENCODER_EPOCHS,
    LOG_VRAM,
    MAX_GRAD_NORM,
    ONECYCLE_MAX_LR,
    PRETRAINED_SWIN_PATH,
    PRETRAINED_SWIN_URL,
    SCHEDULER,
    TRAIN_NUM_SAMPLES,
    USE_DEEP_SUPERVISION,
    USE_EMA,
    USE_FOCAL_LOSS,
    USE_PRETRAINED_SWIN,
    VAL_OVERLAP,
    VAL_SW_BATCH_SIZE,
    WEIGHT_DECAY,
    Config,
    build_loss,
    build_optimizer,
    build_scheduler,
    ensure_experiment_dirs,
)
from datasets.dataloader import get_train_loader, get_val_loader
from models.model_factory import build_model, model_metadata
from models.swinunetr import set_swin_encoder_trainable, verify_encoder_freeze
from utils.brats_metrics import compute_region_dice, logits_to_prediction
from utils.checkpoint_utils import save_checkpoint, try_resume
from utils.ema import ModelEMA
from utils.experiment_logger import ExperimentLogger, MetricRow
from utils.vram_utils import (
    handle_cuda_oom,
    is_cuda_oom,
    print_gpu_startup_banner,
    print_vram_epoch,
    reset_peak_stats,
    warn_if_near_limit,
)


def _make_scaler(enabled: bool):
    """Modern GradScaler when available; fall back for older torch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):  # pragma: no cover
        from torch.cuda.amp import GradScaler as LegacyScaler

        return LegacyScaler(enabled=enabled)


def set_seed(seed: int, *, cudnn_deterministic: bool, cudnn_benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["image"], batch["label"]
    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        return batch[0], batch[1]
    raise ValueError(f"Unexpected batch type: {type(batch)}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Any,
    use_amp: bool,
    *,
    accumulation_steps: int,
    max_grad_norm: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scheduler_is_onecycle: bool,
    ema: Optional[ModelEMA],
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)
    steps = max(1, int(accumulation_steps))
    n_batches = len(loader)
    amp_on = bool(use_amp and device.type == "cuda")

    try:
        for batch_idx, batch in enumerate(loader, start=1):
            x, y = _unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if y.ndim == 5 and int(y.shape[1]) == 1:
                y = y.squeeze(1)

            with torch.amp.autocast("cuda", enabled=amp_on):
                logits = model(x)
                loss = loss_fn(logits, y)
                loss = loss / float(steps)

            scaler.scale(loss).backward()
            # Scalar only — do not retain graph refs
            total_loss += float(loss.item()) * float(steps)
            num_batches += 1

            # Drop references ASAP (graph freed after backward)
            del logits, loss, x, y

            if (batch_idx % steps == 0) or (batch_idx == n_batches):
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler_is_onecycle:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

    except RuntimeError as e:
        if is_cuda_oom(e):
            optimizer.zero_grad(set_to_none=True)
            handle_cuda_oom("training step")
        raise

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
) -> dict[str, float]:
    """Full-volume sliding-window validation. No TTA. sw_batch_size forced to 1."""
    model.eval()
    sum_wt = sum_tc = sum_et = sum_mean = 0.0
    total_samples = 0
    amp_on = bool(use_amp and device.type == "cuda")
    sw_batch_size = 1  # 6GB-safe default

    try:
        for batch in loader:
            x, y = _unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if y.ndim == 5 and int(y.shape[1]) == 1:
                y = y.squeeze(1)

            with torch.amp.autocast("cuda", enabled=amp_on):
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=tuple(int(v) for v in roi_size),
                    sw_batch_size=int(sw_batch_size),
                    predictor=model,
                    overlap=float(overlap),
                )

            pred = logits_to_prediction(logits)
            # Metrics as Python floats (CPU scalars) — do not stash GPU tensors
            scores = compute_region_dice(pred, y, from_logits=False)
            bs = int(x.shape[0])
            sum_wt += float(scores.wt) * bs
            sum_tc += float(scores.tc) * bs
            sum_et += float(scores.et) * bs
            sum_mean += float(scores.mean) * bs
            total_samples += bs

            del logits, pred, x, y, scores
    except RuntimeError as e:
        if is_cuda_oom(e):
            handle_cuda_oom("validation")
        raise

    if total_samples == 0:
        return {"wt_dice": 0.0, "tc_dice": 0.0, "et_dice": 0.0, "mean_dice": 0.0}

    return {
        "wt_dice": sum_wt / total_samples,
        "tc_dice": sum_tc / total_samples,
        "et_dice": sum_et / total_samples,
        "mean_dice": sum_mean / total_samples,
    }


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int,
    use_amp: bool,
    checkpoint_dir: Path,
    log_dir: Path,
    run_metadata: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    *,
    val_roi_size: tuple[int, int, int] = (96, 96, 96),
    val_overlap: float = VAL_OVERLAP,
    val_sw_batch_size: int = VAL_SW_BATCH_SIZE,
    accumulation_steps: int = ACCUMULATION_STEPS,
    max_grad_norm: float = MAX_GRAD_NORM,
    freeze_encoder_epochs: int = FREEZE_ENCODER_EPOCHS,
    use_ema: bool = USE_EMA,
    ema_decay: float = 0.999,
    ema_device: str = EMA_DEVICE,
    start_epoch: int = 1,
    best_scores: Optional[dict[str, float]] = None,
    ema: Optional[ModelEMA] = None,
    model_name: str = "swinunetr",
    scaler: Optional[Any] = None,
    log_vram: bool = LOG_VRAM,
) -> None:
    model.to(device)
    loss_fn.to(device)
    if scaler is None:
        scaler = _make_scaler(enabled=use_amp and device.type == "cuda")
    logger = ExperimentLogger(log_dir=log_dir, filename="training_metrics.csv")

    if best_scores is None:
        best_scores = {
            "mean_dice": float("-inf"),
            "wt_dice": float("-inf"),
            "tc_dice": float("-inf"),
            "et_dice": float("-inf"),
        }

    if use_ema and ema is None:
        ema = ModelEMA(model, decay=ema_decay, device=ema_device)

    scheduler_is_onecycle = isinstance(scheduler, OneCycleLR)
    unfreeze_printed = False

    paths = {
        "last": checkpoint_dir / "last.pt",
        "mean": checkpoint_dir / "best_mean_dice.pt",
        "wt": checkpoint_dir / "best_wt.pt",
        "tc": checkpoint_dir / "best_tc.pt",
        "et": checkpoint_dir / "best_et.pt",
    }

    for epoch in range(int(start_epoch), int(num_epochs) + 1):
        if device.type == "cuda":
            reset_peak_stats()

        if str(model_name).lower() == "swinunetr" and freeze_encoder_epochs > 0:
            if epoch <= int(freeze_encoder_epochs):
                set_swin_encoder_trainable(model, trainable=False)
                if epoch == 1:
                    verify_encoder_freeze(model, expect_frozen=True)
            else:
                set_swin_encoder_trainable(model, trainable=True)
                if not unfreeze_printed:
                    print(f"SwinUNETR encoder unfrozen at epoch {int(freeze_encoder_epochs) + 1}")
                    verify_encoder_freeze(model, expect_frozen=False)
                    unfreeze_printed = True

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            scheduler_is_onecycle=scheduler_is_onecycle,
            ema=ema,
        )

        # Validation uses EMA weights (temporarily on GPU if EMA is CPU-resident).
        # TTA is intentionally NOT used during training validation.
        if use_ema and ema is not None:
            with ema.eval_context(device) as eval_model:
                scores = validate(
                    model=eval_model,
                    loader=val_loader,
                    device=device,
                    use_amp=use_amp,
                    roi_size=val_roi_size,
                    overlap=val_overlap,
                    sw_batch_size=val_sw_batch_size,
                )
        else:
            scores = validate(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                roi_size=val_roi_size,
                overlap=val_overlap,
                sw_batch_size=val_sw_batch_size,
            )

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(scores["mean_dice"])

        current_lr = float(optimizer.param_groups[0]["lr"])
        logger.log(
            MetricRow(
                epoch=epoch,
                train_loss=float(train_loss),
                learning_rate=current_lr,
                wt_dice=float(scores["wt_dice"]),
                tc_dice=float(scores["tc_dice"]),
                et_dice=float(scores["et_dice"]),
                mean_dice=float(scores["mean_dice"]),
            )
        )

        print(
            f"Epoch {epoch:03d}/{num_epochs} | loss={train_loss:.6f} | "
            f"WT={scores['wt_dice']:.4f} TC={scores['tc_dice']:.4f} "
            f"ET={scores['et_dice']:.4f} mean={scores['mean_dice']:.4f} | lr={current_lr:.6g}"
        )
        if log_vram and device.type == "cuda":
            snap = print_vram_epoch(True)
            warn_if_near_limit(snap["peak_allocated_mb"] * (1024.0 ** 2))

        def _meta(tag: str) -> dict[str, Any]:
            m = dict(run_metadata)
            m.update(
                {
                    "checkpoint": tag,
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "learning_rate": float(current_lr),
                    **{k: float(v) for k, v in scores.items()},
                    "best_scores": {k: float(v) for k, v in best_scores.items()},
                    "ema_enabled": bool(use_ema),
                    "ema_device": str(ema_device),
                }
            )
            return m

        save_checkpoint(
            paths["last"],
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
            scheduler=scheduler,
            best_scores=best_scores,
            ema=ema,
            metadata=_meta("last"),
            config_snapshot=config_snapshot,
        )

        improved = []
        if scores["mean_dice"] > best_scores["mean_dice"]:
            best_scores["mean_dice"] = float(scores["mean_dice"])
            improved.append(("mean", paths["mean"], "best_mean_dice"))
        if scores["wt_dice"] > best_scores["wt_dice"]:
            best_scores["wt_dice"] = float(scores["wt_dice"])
            improved.append(("wt", paths["wt"], "best_wt"))
        if scores["tc_dice"] > best_scores["tc_dice"]:
            best_scores["tc_dice"] = float(scores["tc_dice"])
            improved.append(("tc", paths["tc"], "best_tc"))
        if scores["et_dice"] > best_scores["et_dice"]:
            best_scores["et_dice"] = float(scores["et_dice"])
            improved.append(("et", paths["et"], "best_et"))

        for _, path, tag in improved:
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                scaler=scaler,
                scheduler=scheduler,
                best_scores=best_scores,
                ema=ema,
                metadata=_meta(tag),
                config_snapshot=config_snapshot,
            )
            print(f"New best saved: {path.name}")

    plot_paths = logger.plot_curves()
    if plot_paths:
        print(f"Saved training plots to: {log_dir}")


def _write_run_config(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 4-class BraTS SwinUNETR")
    parser.add_argument("--exp_name", type=str, default=EXP_NAME)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/last.pt")
    args = parser.parse_args()

    cfg = Config()
    exp_root = ensure_experiment_dirs(args.exp_name)
    checkpoint_dir = exp_root / "checkpoints"
    log_dir = exp_root / "logs"
    metrics_dir = exp_root / "metrics"

    set_seed(cfg.seed, cudnn_deterministic=cfg.cudnn_deterministic, cudnn_benchmark=cfg.cudnn_benchmark)

    # ---- GPU banner + empty_cache once (not every iteration) ----
    print_gpu_startup_banner()
    print(
        f"[VRAM] AMP={cfg.use_mixed_precision} | checkpointing={cfg.swin_use_checkpoint} | "
        f"TRAIN_NUM_SAMPLES={TRAIN_NUM_SAMPLES} | EMA_DEVICE={EMA_DEVICE} | "
        f"pretrained={USE_PRETRAINED_SWIN}"
    )

    train_loader = get_train_loader(
        root_dir=cfg.train_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        patch_size=cfg.patch_size,
        pos=CROP_POS,
        neg=CROP_NEG,
        num_samples=int(TRAIN_NUM_SAMPLES),
    )
    val_loader = get_val_loader(
        root_dir=cfg.val_dir,
        batch_size=1,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        patch_size=cfg.patch_size,
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, int(cfg.accumulation_steps))))

    freeze_at_start = str(cfg.model_name).lower() == "swinunetr" and int(cfg.freeze_encoder_epochs) > 0

    model, pretrained_report = build_model(
        cfg.model_name,
        in_channels=cfg.input_channels,
        out_channels=cfg.num_classes,
        patch_size=cfg.patch_size,
        baseline_features=cfg.baseline_unet_features,
        residual_features=cfg.residual_unet_features,
        swin_feature_size=cfg.swin_feature_size,
        swin_use_checkpoint=cfg.swin_use_checkpoint,
        use_pretrained_swin=bool(USE_PRETRAINED_SWIN and cfg.model_name.lower() == "swinunetr"),
        swin_pretrained_path=str(PRETRAINED_SWIN_PATH),
        swin_pretrained_url=PRETRAINED_SWIN_URL,
        freeze_swin_encoder=False,
    )

    pretrained_ok = bool(pretrained_report is not None and pretrained_report.loaded)
    if freeze_at_start and pretrained_ok:
        n = set_swin_encoder_trainable(model, trainable=False)
        print(f"[train] Froze {n} Swin encoder params for first {cfg.freeze_encoder_epochs} epochs")
        verify_encoder_freeze(model, expect_frozen=True)
    elif freeze_at_start and not pretrained_ok:
        print(
            "[train] WARNING: FREEZE_ENCODER_EPOCHS>0 but pretrained disabled/failed — "
            "encoder will NOT be frozen (training from scratch)."
        )
        freeze_at_start = False

    effective_freeze_epochs = int(cfg.freeze_encoder_epochs) if freeze_at_start else 0

    model = model.to(cfg.device)
    if cfg.device.type == "cuda":
        torch.cuda.empty_cache()

    optimizer = build_optimizer(model, lr=cfg.learning_rate)
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=steps_per_epoch,
        num_epochs=cfg.num_epochs,
        name=SCHEDULER,
    )
    loss_fn = build_loss()
    scaler = _make_scaler(enabled=cfg.use_mixed_precision and cfg.device.type == "cuda")

    ema = (
        ModelEMA(model, decay=cfg.ema_decay, device=EMA_DEVICE)
        if cfg.use_ema
        else None
    )
    if ema is not None:
        print(f"[EMA] shadow weights on device={EMA_DEVICE}")

    start_epoch = 1
    best_scores = {
        "mean_dice": float("-inf"),
        "wt_dice": float("-inf"),
        "tc_dice": float("-inf"),
        "et_dice": float("-inf"),
    }
    if args.resume:
        start_epoch, best_scores = try_resume(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            expected_classes=cfg.num_classes,
            ema=ema,
        )

    pretrained_record: Any
    if USE_PRETRAINED_SWIN is False:
        pretrained_record = False
    elif pretrained_report is None:
        pretrained_record = False
    else:
        pretrained_record = pretrained_report.as_dict()

    config_snapshot: dict[str, Any] = {
        "exp_name": args.exp_name,
        "num_classes": int(cfg.num_classes),
        "class_mapping": {"0": "background", "1": "NCR/NET", "2": "edema", "3": "ET"},
        "ce_class_weights": [float(v) for v in CE_CLASS_WEIGHTS],
        "model": dict(
            model_metadata(
                cfg.model_name,
                baseline_features=cfg.baseline_unet_features,
                residual_features=cfg.residual_unet_features,
                swin_feature_size=cfg.swin_feature_size,
                swin_use_checkpoint=cfg.swin_use_checkpoint,
            )
        ),
        "deep_supervision": "NOT IMPLEMENTED / FUTURE EXPERIMENT",
        "use_deep_supervision": bool(USE_DEEP_SUPERVISION),
        "pretrained": pretrained_record,
        "freeze_encoder_epochs": int(effective_freeze_epochs),
        "freeze_encoder_requested": int(cfg.freeze_encoder_epochs),
        "freeze_encoder_active": bool(freeze_at_start),
        "training": {
            "epochs": int(cfg.num_epochs),
            "learning_rate": float(cfg.learning_rate),
            "mixed_precision": bool(cfg.use_mixed_precision),
            "batch_size": int(cfg.batch_size),
            "accumulation_steps": int(cfg.accumulation_steps),
            "effective_batch_size": int(cfg.batch_size) * int(cfg.accumulation_steps),
            "patch_size": [int(v) for v in cfg.patch_size],
            "train_num_samples": int(TRAIN_NUM_SAMPLES),
            "seed": int(cfg.seed),
            "device": str(cfg.device),
            "cudnn_deterministic": bool(cfg.cudnn_deterministic),
            "cudnn_benchmark": bool(cfg.cudnn_benchmark),
            "train_dir": str(cfg.train_dir),
            "val_dir": str(cfg.val_dir),
            "crop_pos": int(CROP_POS),
            "crop_neg": int(CROP_NEG),
        },
        "optimizer": {"type": "AdamW", "weight_decay": float(WEIGHT_DECAY)},
        "scheduler": {
            "type": SCHEDULER,
            "max_lr": float(ONECYCLE_MAX_LR),
            "steps_per_epoch": int(steps_per_epoch),
        },
        "loss": {
            "type": "DiceFocalLoss" if USE_FOCAL_LOSS else "DiceCrossEntropyLoss",
            "use_focal_loss": bool(USE_FOCAL_LOSS),
            "focal_gamma": float(FOCAL_GAMMA),
            "ce_class_weights": [float(v) for v in CE_CLASS_WEIGHTS],
        },
        "ema": {
            "enabled": bool(cfg.use_ema),
            "decay": float(cfg.ema_decay),
            "device": str(EMA_DEVICE),
        },
        "validation": {
            "roi": list(cfg.patch_size),
            "overlap": float(cfg.val_overlap),
            "sw_batch_size": int(VAL_SW_BATCH_SIZE),
            "tta_during_training": False,
        },
        "vram": {"log_vram": bool(LOG_VRAM), "ema_device": str(EMA_DEVICE)},
    }
    _write_run_config(metrics_dir / "run_config.json", config_snapshot)
    _write_run_config(exp_root / "run_config.json", config_snapshot)

    run_metadata: dict[str, Any] = {
        "architecture_type": model.__class__.__name__,
        "exp_name": args.exp_name,
        "model": config_snapshot["model"],
        "loss": config_snapshot["loss"],
        "scheduler": config_snapshot["scheduler"],
        "ema": config_snapshot["ema"],
    }

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=cfg.device,
        num_epochs=cfg.num_epochs,
        use_amp=cfg.use_mixed_precision,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        run_metadata=run_metadata,
        config_snapshot=config_snapshot,
        val_roi_size=cfg.patch_size,
        val_overlap=cfg.val_overlap,
        val_sw_batch_size=VAL_SW_BATCH_SIZE,
        accumulation_steps=cfg.accumulation_steps,
        max_grad_norm=MAX_GRAD_NORM,
        freeze_encoder_epochs=effective_freeze_epochs,
        use_ema=cfg.use_ema,
        ema_decay=cfg.ema_decay,
        ema_device=EMA_DEVICE,
        start_epoch=start_epoch,
        best_scores=best_scores,
        ema=ema,
        model_name=cfg.model_name,
        scaler=scaler,
        log_vram=LOG_VRAM,
    )


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.inferers import sliding_window_inference

from configs.config import (
    CE_CLASS_WEIGHTS,
    FOCAL_GAMMA,
    USE_FOCAL_LOSS,
    WEIGHT_DECAY,
    Config,
    build_loss,
    build_optimizer,
)
from datasets.dataloader import get_train_loader, get_val_loader
from models.model_factory import build_model, model_metadata
from utils.experiment_logger import ExperimentLogger, MetricRow
from utils.metrics import multiclass_dice_score_3d


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        x = batch["image"]
        y = batch["label"]
        return x, y
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
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().item())
        num_batches += 1

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
) -> float:
    model.eval()
    total_dice = 0.0
    total_samples = 0

    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=use_amp and device.type == "cuda"):
            logits = sliding_window_inference(
                inputs=x,
                roi_size=tuple(int(v) for v in roi_size),
                sw_batch_size=int(sw_batch_size),
                predictor=model,
                overlap=float(overlap),
            )

        batch_size = int(x.shape[0])
        dice = multiclass_dice_score_3d(logits, y)
        total_dice += dice * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return total_dice / total_samples


def _save_checkpoint_metadata_json(path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_dice": float(best_dice),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        str(path),
    )
    if metadata is not None:
        meta_path = path.with_suffix(".json")
        _save_checkpoint_metadata_json(meta_path, dict(metadata))


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int,
    use_amp: bool,
    checkpoint_dir: Path,
    log_dir: Path,
    run_metadata: Mapping[str, Any],
    val_roi_size: tuple[int, int, int] = (96, 96, 96),
) -> None:
    model.to(device)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    logger = ExperimentLogger(log_dir=log_dir, filename="metrics.csv")

    best_dice = float("-inf")
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_dice = validate(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            roi_size=val_roi_size,
            overlap=0.5,
        )

        scheduler.step(val_dice)

        current_lr = float(optimizer.param_groups[0]["lr"])

        logger.log(
            MetricRow(
                epoch=epoch,
                train_loss=train_loss,
                val_dice=val_dice,
                learning_rate=current_lr,
            )
        )

        print(
            f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.6f} | val_dice={val_dice:.6f} | lr={current_lr:.6f}"
        )

        last_meta = dict(run_metadata)
        last_meta.update(
            {
                "checkpoint": "last",
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_dice": float(val_dice),
                "learning_rate": float(current_lr),
                "best_val_dice_so_far": float(best_dice),
            }
        )
        save_checkpoint(last_path, model=model, optimizer=optimizer, epoch=epoch, best_dice=best_dice, metadata=last_meta)

        if val_dice > best_dice:
            best_dice = val_dice
            best_meta = dict(run_metadata)
            best_meta.update(
                {
                    "checkpoint": "best",
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_dice": float(val_dice),
                    "learning_rate": float(current_lr),
                    "best_val_dice_so_far": float(best_dice),
                }
            )
            save_checkpoint(best_path, model=model, optimizer=optimizer, epoch=epoch, best_dice=best_dice, metadata=best_meta)
            print("New best model saved")

    plot_paths = logger.plot_curves()
    if plot_paths:
        print(f"Saved training plots to: {str(log_dir)}")


def main() -> None:
    cfg = Config()
    seed = int(getattr(cfg, "seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = get_train_loader(
        root_dir=cfg.raw_data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        patch_size=cfg.patch_size,
    )
    val_loader = get_val_loader(
        root_dir=cfg.raw_data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        patch_size=cfg.patch_size,
    )

    model = build_model(
        cfg.model_name,
        in_channels=cfg.input_channels,
        out_channels=cfg.num_classes,
        patch_size=cfg.patch_size,
        baseline_features=cfg.baseline_unet_features,
        residual_features=cfg.residual_unet_features,
        swin_feature_size=cfg.swin_feature_size,
        swin_use_checkpoint=cfg.swin_use_checkpoint,
    )
    optimizer = build_optimizer(model, lr=cfg.learning_rate)
    scheduler_mode = "max"
    scheduler_factor = 0.5
    scheduler_patience = 5
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=scheduler_factor,
        patience=scheduler_patience,
    )
    loss_fn = build_loss()

    run_metadata: dict[str, Any] = {
        "architecture_type": model.__class__.__name__,
        "model": dict(
            model_metadata(
                cfg.model_name,
                baseline_features=cfg.baseline_unet_features,
                residual_features=cfg.residual_unet_features,
                swin_feature_size=cfg.swin_feature_size,
                swin_use_checkpoint=cfg.swin_use_checkpoint,
            )
        ),
        "training": {
            "epochs": int(cfg.num_epochs),
            "learning_rate": float(cfg.learning_rate),
            "mixed_precision": bool(cfg.use_mixed_precision),
            "batch_size": int(cfg.batch_size),
            "patch_size": [int(v) for v in cfg.patch_size],
            "seed": int(seed),
            "device": str(cfg.device),
        },
        "optimizer": {
            "type": optimizer.__class__.__name__,
            "weight_decay": float(WEIGHT_DECAY),
        },
        "scheduler": {
            "type": scheduler.__class__.__name__,
            "mode": str(scheduler_mode),
            "factor": float(scheduler_factor),
            "patience": int(scheduler_patience),
        },
        "loss": {
            "type": "DiceFocalLoss" if USE_FOCAL_LOSS else "DiceCrossEntropyLoss",
            "use_focal_loss": bool(USE_FOCAL_LOSS),
            "focal_gamma": float(FOCAL_GAMMA),
            "ce_class_weights": [float(v) for v in CE_CLASS_WEIGHTS],
        },
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
        checkpoint_dir=cfg.checkpoint_dir,
        log_dir=cfg.log_dir,
        run_metadata=run_metadata,
        val_roi_size=cfg.patch_size,
    )


if __name__ == "__main__":
    main()


from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from configs.config import Config, build_loss, build_optimizer
from datasets.dataloader import get_train_loader, get_val_loader
from models.unet3d import UNet3D
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
) -> float:
    model.eval()
    total_dice = 0.0
    total_samples = 0

    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(x)

        batch_size = int(x.shape[0])
        dice = multiclass_dice_score_3d(logits, y)
        total_dice += dice * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return total_dice / total_samples


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_dice: float) -> None:
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


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int,
    use_amp: bool,
    checkpoint_dir: Path,
) -> None:
    model.to(device)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    best_dice = float("-inf")
    best_path = checkpoint_dir / "best.pt"

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

        val_dice = validate(model=model, loader=val_loader, device=device, use_amp=use_amp)

        print(f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.6f} | val_dice={val_dice:.6f}")

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(best_path, model=model, optimizer=optimizer, epoch=epoch, best_dice=best_dice)


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

    model = UNet3D(in_channels=cfg.input_channels, out_channels=cfg.num_classes)
    optimizer = build_optimizer(model, lr=cfg.learning_rate)
    loss_fn = build_loss()

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=cfg.device,
        num_epochs=cfg.num_epochs,
        use_amp=cfg.use_mixed_precision,
        checkpoint_dir=cfg.checkpoint_dir,
    )


if __name__ == "__main__":
    main()

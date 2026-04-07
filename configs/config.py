
"""Project configuration for 3D brain tumor segmentation (PyTorch).

This module keeps all experiment settings in one place:
- Device selection (CUDA if available)
- Dataset paths
- Model and training hyperparameters
- Optimizer and loss factory helpers
- Output directories (checkpoints/logs)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """Return the default compute device."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# -------------------------------
# Core runtime settings
# -------------------------------

# Reproducibility seed (set in your training script with torch.manual_seed(SEED)).
SEED: int = 42

DEVICE: torch.device = get_device()

# Enable mixed precision training (use torch.cuda.amp/autocast in your training loop).
USE_MIXED_PRECISION: bool = True

# Checkpoint behavior.
SAVE_BEST_ONLY: bool = True


# -------------------------------
# Dataset settings
# -------------------------------

# Project root = two levels up from this file: <root>/configs/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Update these paths to match your local dataset layout.
DATA_ROOT: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_ROOT / "raw"
PROCESSED_DATA_DIR: Path = DATA_ROOT / "processed"

# MRI modalities used for BraTS-style inputs.
MODALITIES: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")
INPUT_CHANNELS: int = 4


# -------------------------------
# DataLoader performance settings
# -------------------------------

# For Windows, use a small number of workers to avoid excessive overhead.
NUM_WORKERS: int = 2
PIN_MEMORY: bool = True


# -------------------------------
# Model / training hyperparameters
# -------------------------------

# Patch size chosen to fit common 6GB GPUs for 3D U-Net style models.
PATCH_SIZE: tuple[int, int, int] = (96, 96, 96)

BATCH_SIZE: int = 1
NUM_EPOCHS: int = 100
LEARNING_RATE: float = 1e-4

# Number of classes after label remapping (e.g., WT/TC/ET).
NUM_CLASSES: int = 3

# AdamW optimizer settings.
WEIGHT_DECAY: float = 1e-2


# -------------------------------
# Output directories
# -------------------------------

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
LOG_DIR: Path = OUTPUT_DIR / "logs"


# -------------------------------
# Loss: Dice + CrossEntropy
# -------------------------------


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss.

    Notes:
    - Expects logits shaped (N, C, D, H, W)
    - Expects targets shaped (N, D, H, W) with class indices in [0, C-1]
    """

    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_1h = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        target_1h = target_1h.to(dtype=probs.dtype)

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_1h, dims)
        denom = torch.sum(probs + target_1h, dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        # Average across classes.
        return 1.0 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    """Combined Dice + CrossEntropy loss."""

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes, smooth=smooth)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target)


def build_optimizer(model: nn.Module, lr: float = LEARNING_RATE) -> torch.optim.Optimizer:
    """Create the AdamW optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)


def build_loss() -> nn.Module:
    """Create the Dice + CrossEntropy loss."""
    return DiceCrossEntropyLoss(num_classes=NUM_CLASSES)


# -------------------------------
# Optional: a single structured object for convenience
# -------------------------------


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    device: torch.device = DEVICE
    use_mixed_precision: bool = USE_MIXED_PRECISION
    save_best_only: bool = SAVE_BEST_ONLY

    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR

    modalities: tuple[str, ...] = MODALITIES
    input_channels: int = INPUT_CHANNELS
    patch_size: tuple[int, int, int] = PATCH_SIZE

    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    pin_memory: bool = PIN_MEMORY
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    num_classes: int = NUM_CLASSES

    output_dir: Path = OUTPUT_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    log_dir: Path = LOG_DIR


"""Project configuration for 3D brain tumor segmentation (PyTorch).

Single source of experiment settings for the final 4-class SwinUNETR run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import FocalLoss


def get_device() -> torch.device:
    """Return the default compute device."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# -------------------------------
# Core runtime settings
# -------------------------------

SEED: int = 42
DEVICE: torch.device = get_device()
USE_MIXED_PRECISION: bool = True
SAVE_BEST_ONLY: bool = True

# Determinism (recorded in run_config.json)
CUDNN_DETERMINISTIC: bool = True
CUDNN_BENCHMARK: bool = False

USE_FOCAL_LOSS: bool = False
FOCAL_GAMMA: float = 2.0

# Deep supervision = NOT IMPLEMENTED / FUTURE EXPERIMENT
USE_DEEP_SUPERVISION: bool = False


# -------------------------------
# Dataset settings
# -------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_ROOT: Path = PROJECT_ROOT.parent / "BraTS"
TRAIN_DIR: Path = DATA_ROOT / "Training"
VAL_DIR: Path = DATA_ROOT / "Validation"
TEST_DIR: Path = DATA_ROOT / "Testing"

# MRI modalities: t1n, t1c, t2w, t2f
MODALITIES: tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")
INPUT_CHANNELS: int = 4


# -------------------------------
# DataLoader performance settings
# -------------------------------

# Windows: 0 avoids DataLoader worker hangs; increase on Linux if desired.
NUM_WORKERS: int = 0
PIN_MEMORY: bool = True


# -------------------------------
# Model / training hyperparameters
# -------------------------------

PATCH_SIZE: tuple[int, int, int] = (96, 96, 96)

# Model selection: baseline_unet | residual_unet | swinunetr
MODEL_NAME: str = "swinunetr"

BASELINE_UNET_FEATURES: tuple[int, int, int, int] = (16, 32, 64, 128)
RESIDUAL_UNET_FEATURES: tuple[int, int, int, int] = (32, 64, 128, 256)

SWIN_FEATURE_SIZE: int = 24
SWIN_USE_CHECKPOINT: bool = True

# Official SSL SwinViT weights are feature_size=48 — incompatible with feature_size=24.
# Disabled for this 6GB experiment (avoids unused 392MB download; train from scratch).
USE_PRETRAINED_SWIN: bool = False
PRETRAINED_SWIN_PATH: Path = PROJECT_ROOT / "pretrained" / "model_swinvit.pt"
PRETRAINED_SWIN_URL: str = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
)
# Freeze Swin encoder for the first N epochs (only meaningful if pretrained loaded).
FREEZE_ENCODER_EPOCHS: int = 30

BATCH_SIZE: int = 1
NUM_EPOCHS: int = 75
LEARNING_RATE: float = 1e-4

# 0=background, 1=NCR/NET, 2=edema, 3=enhancing tumor (ET; BraTS label 4 remapped)
NUM_CLASSES: int = 4
CE_CLASS_WEIGHTS: tuple[float, ...] = (0.1, 1.0, 2.0, 4.0)

WEIGHT_DECAY: float = 1e-2

# Scheduler: "onecycle" (default) or "plateau"
SCHEDULER: str = "onecycle"
ONECYCLE_MAX_LR: float = 1e-3
ONECYCLE_PCT_START: float = 0.1
ONECYCLE_ANNEAL_STRATEGY: str = "cos"
PLATEAU_FACTOR: float = 0.5
PLATEAU_PATIENCE: int = 5

# Gradient accumulation: physical batch=1 → effective batch ≈ ACCUMULATION_STEPS
ACCUMULATION_STEPS: int = 4
MAX_GRAD_NORM: float = 1.0

# Tumor-centered patch sampling (train only).
# MONAI RandCropByPosNegLabeld(num_samples=N) + list_data_collate expands each
# case into N patches → model sees shape [N, 4, 96, 96, 96] when BATCH_SIZE=1.
# Preferred: 4 → 2 → 1 only if VRAM dry-run requires it.
CROP_POS: int = 3
CROP_NEG: int = 1
TRAIN_NUM_SAMPLES: int = 2  # reduced from 4: dry-run peak ~8GB at 4 samples on RTX 4050 6GB; 2 samples ~4.2GB
NUM_SAMPLES_PER_VOLUME: int = TRAIN_NUM_SAMPLES  # back-compat alias

# EMA — keep shadow weights on CPU to avoid a second full SwinUNETR on the 6GB GPU.
USE_EMA: bool = True
EMA_DECAY: float = 0.999
EMA_DEVICE: str = "cpu"

# Validation / inference
VAL_OVERLAP: float = 0.5
VAL_SW_BATCH_SIZE: int = 1
USE_TTA: bool = True  # inference/eval only — never during training validation

# VRAM monitoring (epoch-end print). Disable with LOG_VRAM=False.
LOG_VRAM: bool = True
VRAM_WARN_GB: float = 5.5
VRAM_CRITICAL_GB: float = 5.8

# Optional connected-component post-processing (inference only; raw preds untouched)
USE_CC_POSTPROCESS: bool = False
CC_MIN_SIZE: int = 50  # voxels; conservative — does NOT keep-only-largest

LABEL_SMOOTHING: float = 0.0


# -------------------------------
# Experiment output directories
# -------------------------------

EXP_NAME: str = "exp_swinunetr_4class_final"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
EXP_DIR: Path = OUTPUT_DIR / EXP_NAME
CHECKPOINT_DIR: Path = EXP_DIR / "checkpoints"
LOG_DIR: Path = EXP_DIR / "logs"
METRICS_DIR: Path = EXP_DIR / "metrics"
PREDICTIONS_DIR: Path = EXP_DIR / "predictions"
VISUALIZATIONS_DIR: Path = EXP_DIR / "visualizations"


def ensure_experiment_dirs(exp_name: str | None = None) -> Path:
    """Create and return the experiment root directory (does not touch other exps)."""
    name = EXP_NAME if exp_name is None else str(exp_name)
    root = OUTPUT_DIR / name
    for sub in ("checkpoints", "logs", "metrics", "predictions", "visualizations"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


# -------------------------------
# Loss: Dice + CrossEntropy
# -------------------------------


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss.

    Expects:
      logits  [N, C, D, H, W]
      targets [N, D, H, W] long class indices in [0, C-1]
    """

    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        if target.ndim == 5 and int(target.shape[1]) == 1:
            target = target.squeeze(1)
        target_1h = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        target_1h = target_1h.to(dtype=probs.dtype)

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_1h, dims)
        denom = torch.sum(probs + target_1h, dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    """Combined Dice + weighted CrossEntropy (AMP-safe; CE weights follow logits.device)."""

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        ce_class_weights: tuple[float, ...] | None = None,
        smooth: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.dice = DiceLoss(num_classes=num_classes, smooth=smooth)
        self._has_ce_weights = ce_class_weights is not None
        if ce_class_weights is not None:
            if len(ce_class_weights) != int(num_classes):
                raise ValueError(
                    f"Expected ce_class_weights length {int(num_classes)}, got {len(ce_class_weights)}"
                )
            self.register_buffer(
                "ce_class_weights",
                torch.tensor(ce_class_weights, dtype=torch.float32),
            )
        else:
            # Placeholder buffer so .to(device) still works consistently.
            self.register_buffer("ce_class_weights", torch.empty(0))

        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 5 and int(target.shape[1]) == 1:
            target = target.squeeze(1)
        target = target.long()

        weight = None
        if self._has_ce_weights:
            weight = self.ce_class_weights.to(device=logits.device, dtype=torch.float32)

        # CE in float32 for AMP stability; logits may be fp16 under autocast.
        ce = F.cross_entropy(
            logits.float(),
            target,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * ce


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal loss (optional)."""

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_gamma: float = FOCAL_GAMMA,
        focal_class_weights: tuple[float, ...] | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes, smooth=smooth)

        if focal_class_weights is not None:
            if len(focal_class_weights) != int(num_classes):
                raise ValueError(
                    f"Expected focal_class_weights length {int(num_classes)}, got {len(focal_class_weights)}"
                )
            self.register_buffer(
                "focal_class_weights",
                torch.tensor(focal_class_weights, dtype=torch.float32),
            )
        else:
            self.focal_class_weights = None

        self.focal = FocalLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            gamma=float(focal_gamma),
            weight=self.focal_class_weights,
        )
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 5 and int(target.shape[1]) == 1:
            target = target.squeeze(1)
        return self.dice_weight * self.dice(logits, target) + self.focal_weight * self.focal(logits, target)


def build_optimizer(model: nn.Module, lr: float = LEARNING_RATE) -> torch.optim.Optimizer:
    """Create the AdamW optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)


def build_loss() -> nn.Module:
    """Create Dice + weighted CrossEntropy (or Dice+Focal if enabled)."""
    if USE_FOCAL_LOSS:
        return DiceFocalLoss(num_classes=NUM_CLASSES, focal_class_weights=CE_CLASS_WEIGHTS)
    return DiceCrossEntropyLoss(
        num_classes=NUM_CLASSES,
        ce_class_weights=CE_CLASS_WEIGHTS,
        label_smoothing=LABEL_SMOOTHING,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
    num_epochs: int = NUM_EPOCHS,
    name: str | None = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build OneCycleLR (default) or ReduceLROnPlateau fallback."""
    sched = (SCHEDULER if name is None else name).lower().strip()
    if sched == "onecycle":
        total_steps = max(1, int(steps_per_epoch) * int(num_epochs))
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(ONECYCLE_MAX_LR),
            total_steps=total_steps,
            pct_start=float(ONECYCLE_PCT_START),
            anneal_strategy=str(ONECYCLE_ANNEAL_STRATEGY),
        )
    if sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(PLATEAU_FACTOR),
            patience=int(PLATEAU_PATIENCE),
        )
    raise ValueError(f"Unknown SCHEDULER={sched!r}. Expected 'onecycle' or 'plateau'.")


# -------------------------------
# Structured config object
# -------------------------------


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    device: torch.device = DEVICE
    use_mixed_precision: bool = USE_MIXED_PRECISION
    save_best_only: bool = SAVE_BEST_ONLY
    cudnn_deterministic: bool = CUDNN_DETERMINISTIC
    cudnn_benchmark: bool = CUDNN_BENCHMARK

    train_dir: Path = TRAIN_DIR
    val_dir: Path = VAL_DIR
    test_dir: Path = TEST_DIR
    # Back-compat alias used by older scripts
    raw_data_dir: Path = TRAIN_DIR
    processed_data_dir: Path = DATA_ROOT

    modalities: tuple[str, ...] = MODALITIES
    input_channels: int = INPUT_CHANNELS
    patch_size: tuple[int, int, int] = PATCH_SIZE
    model_name: str = MODEL_NAME
    baseline_unet_features: tuple[int, int, int, int] = BASELINE_UNET_FEATURES
    residual_unet_features: tuple[int, int, int, int] = RESIDUAL_UNET_FEATURES
    swin_feature_size: int = SWIN_FEATURE_SIZE
    swin_use_checkpoint: bool = SWIN_USE_CHECKPOINT
    use_pretrained_swin: bool = USE_PRETRAINED_SWIN
    pretrained_swin_path: Path = PRETRAINED_SWIN_PATH
    freeze_encoder_epochs: int = FREEZE_ENCODER_EPOCHS

    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    pin_memory: bool = PIN_MEMORY
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    num_classes: int = NUM_CLASSES
    accumulation_steps: int = ACCUMULATION_STEPS
    max_grad_norm: float = MAX_GRAD_NORM
    scheduler: str = SCHEDULER
    use_ema: bool = USE_EMA
    ema_decay: float = EMA_DECAY
    ema_device: str = EMA_DEVICE
    train_num_samples: int = TRAIN_NUM_SAMPLES
    val_overlap: float = VAL_OVERLAP
    val_sw_batch_size: int = VAL_SW_BATCH_SIZE
    use_tta: bool = USE_TTA
    log_vram: bool = LOG_VRAM

    exp_name: str = EXP_NAME
    output_dir: Path = OUTPUT_DIR
    exp_dir: Path = EXP_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    log_dir: Path = LOG_DIR
    metrics_dir: Path = METRICS_DIR
    predictions_dir: Path = PREDICTIONS_DIR
    visualizations_dir: Path = VISUALIZATIONS_DIR

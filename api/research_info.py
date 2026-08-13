"""Research and model metadata exposed via API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.config import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    CE_CLASS_WEIGHTS,
    EXP_NAME,
    INPUT_CHANNELS,
    LEARNING_RATE,
    MODEL_NAME,
    MODALITIES,
    NUM_CLASSES,
    NUM_EPOCHS,
    PATCH_SIZE,
    SCHEDULER,
    SWIN_FEATURE_SIZE,
    SWIN_USE_CHECKPOINT,
    TRAIN_DIR,
    USE_EMA,
    USE_MIXED_PRECISION,
    USE_TTA,
    VAL_DIR,
    WEIGHT_DECAY,
)


def get_methods_summary() -> dict[str, Any]:
    """Return project configuration for methods/reproducibility display."""
    return {
        "dataset": {
            "name": "BraTS (BraTS-GLI layout)",
            "training_path": str(TRAIN_DIR),
            "validation_path": str(VAL_DIR),
            "subset_split": "Training / Validation folders under local BraTS dataset root",
        },
        "input_modalities": ["T1", "T1ce", "T2", "FLAIR"],
        "internal_modality_keys": list(MODALITIES),
        "preprocessing": {
            "normalization": "Per-volume z-score normalization (inference and training)",
            "resampling": "Not specified",
            "patch_extraction": f"Training: tumor-centered random crops {PATCH_SIZE}; inference: sliding-window over full volume",
            "orientation": "NIfTI canonical orientation",
        },
        "architecture": {
            "name": MODEL_NAME,
            "type": "3D segmentation",
            "output_classes": NUM_CLASSES,
            "input_channels": INPUT_CHANNELS,
            "patch_size": list(PATCH_SIZE),
            "swin_feature_size": SWIN_FEATURE_SIZE,
            "swin_gradient_checkpointing": SWIN_USE_CHECKPOINT,
        },
        "training": {
            "optimizer": "AdamW",
            "loss_function": "Dice + weighted CrossEntropy",
            "ce_class_weights": list(CE_CLASS_WEIGHTS),
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": SCHEDULER,
            "augmentation": "MONAI spatial/intensity transforms (RandCropByPosNegLabeld, flips, rotate90, affine, bias field, contrast, Gaussian noise/smooth, intensity scale/shift)",
            "ema": USE_EMA,
            "mixed_precision": USE_MIXED_PRECISION,
            "tta_inference": USE_TTA,
        },
        "hardware": {
            "gpu": "CUDA when available (project notes reference RTX 4050 6GB for development)",
            "cpu": "Fallback inference on CPU",
            "ram": "Not specified",
        },
        "experiment_name": EXP_NAME,
    }


def get_model_info(checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Return model metadata; optionally include checkpoint validation scores."""
    info: dict[str, Any] = {
        "architecture": "SwinUNETR",
        "task": "3D Multi-Class Brain Tumor Segmentation",
        "input_modalities": ["T1", "T1ce", "T2", "FLAIR"],
        "output_classes": NUM_CLASSES,
        "classes": {
            "0": "Background",
            "1": "NCR/NET",
            "2": "Edema",
            "3": "Enhancing Tumor (ET)",
        },
        "dataset": "BraTS (BraTS-GLI label mapping: 0–3)",
        "checkpoint": str(checkpoint_path) if checkpoint_path else f"outputs/{EXP_NAME}/checkpoints/best_mean_dice.pt",
        "parameter_count": None,
        "training_validation_scores": None,
    }

    try:
        import torch
        from configs.config import PATCH_SIZE as PS
        from models.model_factory import build_model

        model, _ = build_model(
            model_name=MODEL_NAME,
            in_channels=INPUT_CHANNELS,
            out_channels=NUM_CLASSES,
            patch_size=PS,
            swin_feature_size=SWIN_FEATURE_SIZE,
            swin_use_checkpoint=SWIN_USE_CHECKPOINT,
        )
        info["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
        del model
    except Exception:
        info["parameter_count"] = None

    if checkpoint_path and checkpoint_path.exists():
        try:
            import torch

            ckpt = torch.load(str(checkpoint_path), map_location="cpu")
            if isinstance(ckpt, dict) and "best_scores" in ckpt:
                info["training_validation_scores"] = {
                    k: float(v) for k, v in ckpt["best_scores"].items()
                }
                info["training_validation_note"] = (
                    "Scores from held-out validation during model training — "
                    "not metrics for the current uploaded case."
                )
        except Exception:
            pass

    return info

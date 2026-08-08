"""MONAI SwinUNETR wrapper with optional pretrained encoder loading and freeze helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SwinUNETRConfig:
    img_size: tuple[int, int, int] = (96, 96, 96)
    in_channels: int = 4
    out_channels: int = 4

    # Memory/perf tuning (6GB-friendly defaults).
    feature_size: int = 24
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    use_checkpoint: bool = True


@dataclass(frozen=True)
class PretrainedLoadReport:
    loaded: bool
    loaded_keys: int
    skipped_keys: int
    missing_in_checkpoint: int
    skipped_names: tuple[str, ...]
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pretrained_weights_loaded": "YES" if self.loaded else "NO",
            "loaded_keys": int(self.loaded_keys),
            "skipped_keys": int(self.skipped_keys),
            "missing_in_checkpoint": int(self.missing_in_checkpoint),
            "skipped_names_sample": list(self.skipped_names[:40]),
            "message": self.message,
        }


def build_swinunetr(cfg: SwinUNETRConfig) -> nn.Module:
    """
    Build MONAI SwinUNETR for 3D segmentation.

    Notes:
    - `img_size` must match training/inference ROI (96^3).
    - `use_checkpoint=True` reduces VRAM at the cost of speed.
    - Output: [N, out_channels, D, H, W] for input [N, in_channels, D, H, W].
    """
    try:
        from monai.networks.nets import SwinUNETR  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MONAI SwinUNETR is not available. Ensure `monai` is installed with the correct version."
        ) from e

    return SwinUNETR(
        in_channels=int(cfg.in_channels),
        out_channels=int(cfg.out_channels),
        feature_size=int(cfg.feature_size),
        drop_rate=float(cfg.drop_rate),
        attn_drop_rate=float(cfg.attn_drop_rate),
        dropout_path_rate=float(cfg.dropout_path_rate),
        use_checkpoint=bool(cfg.use_checkpoint),
        spatial_dims=3,
    )


def _is_output_head_key(key: str) -> bool:
    k = key.lower()
    return (
        k.startswith("out.")
        or ".out." in k
        or k.endswith("out.conv.conv.weight")
        or k.endswith("out.conv.conv.bias")
        or "out.conv" in k
    )


def _unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "student", "teacher"):
            if key in obj and isinstance(obj[key], dict):
                return dict(obj[key])
        # Already a flat state dict?
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return dict(obj)
    raise ValueError("Unrecognized pretrained checkpoint format (expected state_dict-like mapping).")


def _try_monai_load_from(model: nn.Module, weights_path: Path) -> Optional[PretrainedLoadReport]:
    """Use SwinUNETR.load_from when available (BTCV SSL SwinViT format)."""
    if not hasattr(model, "load_from"):
        return None
    try:
        weights = torch.load(str(weights_path), map_location="cpu")
        # MONAI load_from expects the raw checkpoint dict for model_swinvit.pt
        model.load_from(weights)  # type: ignore[attr-defined]
        # load_from does not return missing/unexpected; report best-effort.
        return PretrainedLoadReport(
            loaded=True,
            loaded_keys=-1,
            skipped_keys=-1,
            missing_in_checkpoint=0,
            skipped_names=("out.* (segmentation head kept randomly initialized by load_from)",),
            message=(
                "Pretrained weights loaded: YES (via model.load_from). "
                "Segmentation head remains freshly initialized for 4 output classes."
            ),
        )
    except Exception as e:
        print(f"[swinunetr] model.load_from failed ({e}); falling back to shape-matched load.")
        return None


def load_pretrained_swin_weights(
    model: nn.Module,
    *,
    weights_path: str | Path,
    download_url: str | None = None,
) -> PretrainedLoadReport:
    """
    Load compatible pretrained encoder weights into SwinUNETR.

    - Skips incompatible shapes (e.g. different in_channels / feature_size).
    - Always skips / never overwrites the final segmentation head (4-class).
    - Does NOT use blind strict=False without reporting.
    """
    path = Path(weights_path)
    if not path.exists():
        if download_url:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[swinunetr] Downloading pretrained weights from {download_url}")
                torch.hub.download_url_to_file(download_url, str(path))
            except Exception as e:
                msg = f"Pretrained weights loaded: NO (download failed: {e})"
                print(msg)
                return PretrainedLoadReport(False, 0, 0, 0, (), msg)
        else:
            msg = f"Pretrained weights loaded: NO (file not found: {path})"
            print(msg)
            return PretrainedLoadReport(False, 0, 0, 0, (), msg)

    # Preferred path for official model_swinvit.pt
    report = _try_monai_load_from(model, path)
    if report is not None and report.loaded:
        print(report.message)
        print(f"Loaded keys: {report.loaded_keys}")
        print(f"Skipped keys: {report.skipped_keys}")
        return report

    raw = torch.load(str(path), map_location="cpu")
    src = _unwrap_state_dict(raw)

    # Remap common SSL prefixes → MONAI SwinUNETR keys
    remapped: dict[str, torch.Tensor] = {}
    for key, value in src.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = key
        if key.startswith("module."):
            new_key = key[len("module.") :]
        if new_key.startswith("encoder."):
            # SSL → swinViT mapping (approximate; shape filter still applies)
            rest = new_key[len("encoder.") :]
            if rest.startswith("patch_embed"):
                new_key = "swinViT." + rest
            else:
                # layersX.0 → layersX (common SSL quirk)
                parts = rest.split(".")
                if len(parts) >= 2 and parts[1] == "0":
                    new_key = "swinViT." + parts[0] + "." + ".".join(parts[2:])
                else:
                    new_key = "swinViT." + rest
        remapped[new_key] = value
        remapped[key] = value  # also keep original key for direct matches

    model_sd = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, tensor in remapped.items():
        if _is_output_head_key(key):
            skipped.append(f"{key} (output head — keep 4-class random init)")
            continue
        if key not in model_sd:
            skipped.append(f"{key} (not in model)")
            continue
        if tuple(tensor.shape) != tuple(model_sd[key].shape):
            skipped.append(
                f"{key} (shape {tuple(tensor.shape)} != model {tuple(model_sd[key].shape)})"
            )
            continue
        loadable[key] = tensor

    missing = [k for k in model_sd.keys() if k not in loadable and not _is_output_head_key(k)]
    # Load only compatible tensors
    result = model.load_state_dict(loadable, strict=False)
    unexpected = list(getattr(result, "unexpected_keys", []))
    missing_keys = list(getattr(result, "missing_keys", []))

    loaded_n = len(loadable)
    skipped_n = len(skipped) + len(unexpected)
    loaded_yes = loaded_n > 0
    msg = (
        f"Pretrained weights loaded: {'YES' if loaded_yes else 'NO'} | "
        f"Loaded keys: {loaded_n} | Skipped keys: {skipped_n} | "
        f"Missing in model after load: {len(missing_keys)}"
    )
    print(msg)
    if skipped:
        print(f"[swinunetr] Skipped incompatible/head keys (showing up to 25):")
        for name in skipped[:25]:
            print(f"  - {name}")
        if len(skipped) > 25:
            print(f"  ... and {len(skipped) - 25} more")
    if missing_keys:
        print(f"[swinunetr] Missing keys after partial load: {len(missing_keys)} (expected for decoder/head)")

    return PretrainedLoadReport(
        loaded=loaded_yes,
        loaded_keys=loaded_n,
        skipped_keys=skipped_n,
        missing_in_checkpoint=len(missing),
        skipped_names=tuple(skipped[:80]),
        message=msg,
    )


def set_swin_encoder_trainable(model: nn.Module, *, trainable: bool) -> int:
    """Freeze/unfreeze parameters whose name contains 'swinViT'. Returns count touched."""
    n = 0
    for name, param in model.named_parameters():
        if "swinViT" in name or "swinvit" in name.lower():
            param.requires_grad = bool(trainable)
            n += 1
    return n


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def verify_encoder_freeze(model: nn.Module, *, expect_frozen: bool) -> bool:
    """Programmatic requires_grad check for Swin encoder params."""
    encoder_params = [
        (n, p) for n, p in model.named_parameters() if "swinViT" in n or "swinvit" in n.lower()
    ]
    if not encoder_params:
        print("[swinunetr] WARNING: no swinViT parameters found for freeze check")
        return False
    ok = all((not p.requires_grad) if expect_frozen else p.requires_grad for _, p in encoder_params)
    # Decoder/head should remain trainable when encoder is frozen
    other = [
        (n, p)
        for n, p in model.named_parameters()
        if not ("swinViT" in n or "swinvit" in n.lower())
    ]
    decoder_ok = all(p.requires_grad for _, p in other) if expect_frozen else True
    print(
        f"[swinunetr] encoder_freeze_ok={ok} expect_frozen={expect_frozen} "
        f"encoder_params={len(encoder_params)} decoder_trainable_ok={decoder_ok}"
    )
    return bool(ok and decoder_ok)

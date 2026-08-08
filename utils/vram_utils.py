"""Lightweight CUDA VRAM reporting helpers for 6GB-safe training."""

from __future__ import annotations

from typing import Any


def bytes_to_mb(n: int | float) -> float:
    return float(n) / (1024.0 ** 2)


def bytes_to_gb(n: int | float) -> float:
    return float(n) / (1024.0 ** 3)


def cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def reset_peak_stats() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def print_gpu_startup_banner() -> dict[str, Any]:
    """Print GPU identity and current VRAM; call once before training."""
    import torch

    info: dict[str, Any] = {"cuda": False}
    if not torch.cuda.is_available():
        print("[VRAM] CUDA not available - training will use CPU")
        return info

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total = int(props.total_memory)
    free, total_mem = torch.cuda.mem_get_info(idx)
    allocated = int(torch.cuda.memory_allocated(idx))
    reserved = int(torch.cuda.memory_reserved(idx))

    info = {
        "cuda": True,
        "name": props.name,
        "total_bytes": total,
        "free_bytes": int(free),
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "total_mb": bytes_to_mb(total),
        "free_mb": bytes_to_mb(free),
        "allocated_mb": bytes_to_mb(allocated),
        "reserved_mb": bytes_to_mb(reserved),
    }
    print("[VRAM] GPU startup")
    print(f"  GPU name:        {props.name}")
    print(f"  total VRAM:      {bytes_to_mb(total):.1f} MB ({bytes_to_gb(total):.2f} GB)")
    print(f"  allocated VRAM:  {bytes_to_mb(allocated):.1f} MB")
    print(f"  reserved VRAM:   {bytes_to_mb(reserved):.1f} MB")
    print(f"  free VRAM:       {bytes_to_mb(free):.1f} MB")
    return info


def vram_snapshot() -> dict[str, float]:
    """Current + peak allocated/reserved in MB."""
    import torch

    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }
    idx = torch.cuda.current_device()
    return {
        "allocated_mb": bytes_to_mb(torch.cuda.memory_allocated(idx)),
        "reserved_mb": bytes_to_mb(torch.cuda.memory_reserved(idx)),
        "peak_allocated_mb": bytes_to_mb(torch.cuda.max_memory_allocated(idx)),
        "peak_reserved_mb": bytes_to_mb(torch.cuda.max_memory_reserved(idx)),
    }


def format_vram_line(snap: dict[str, float] | None = None) -> str:
    s = snap if snap is not None else vram_snapshot()
    return (
        f"VRAM: allocated={s['allocated_mb']:.0f} MB | "
        f"reserved={s['reserved_mb']:.0f} MB | "
        f"peak={s['peak_allocated_mb']:.0f} MB"
    )


def print_vram_epoch(log_vram: bool = True) -> dict[str, float]:
    snap = vram_snapshot()
    if log_vram:
        print(format_vram_line(snap))
    return snap


def classify_peak_gb(peak_allocated_gb: float) -> str:
    """Return SAFE / CLOSE / UNSAFE / CRITICAL relative to 6GB card."""
    if peak_allocated_gb > 5.8:
        return "CRITICAL"
    if peak_allocated_gb > 5.5:
        return "CLOSE"
    if peak_allocated_gb > 5.0:
        return "CAUTION"
    return "SAFE"


def warn_if_near_limit(peak_allocated_bytes: int | float) -> str:
    """Print WARNING/CRITICAL messages for 6GB GPUs. Returns level string."""
    gb = bytes_to_gb(peak_allocated_bytes)
    level = classify_peak_gb(gb)
    if level == "CRITICAL":
        print("CRITICAL: Reduce TRAIN_NUM_SAMPLES before full training.")
    elif level == "CLOSE":
        print("WARNING: VRAM usage is dangerously close to the 6 GB GPU limit.")
    return level


def is_cuda_oom(err: BaseException) -> bool:
    if not isinstance(err, RuntimeError):
        return False
    msg = str(err).lower()
    return ("out of memory" in msg) or ("cuda" in msg and "oom" in msg)


def handle_cuda_oom(context: str = "training") -> None:
    """Clear grads/cache and raise a clear abort (do not retry indefinitely)."""
    import torch

    print(f"[OOM] CUDA out-of-memory during {context}. Aborting run safely.")
    print("[OOM] Clearing gradients and CUDA cache...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    raise SystemExit(
        f"CUDA OOM during {context}. "
        "Try TRAIN_NUM_SAMPLES=2 (then 1), keep EMA on CPU, ensure SWIN_USE_CHECKPOINT=True."
    )

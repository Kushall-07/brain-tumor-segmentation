"""Exponential moving average of model parameters (VRAM-safe CPU residency)."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn


class ModelEMA:
    """
    Shadow copy of model weights, updated after optimizer steps only.

    By default the EMA copy lives on CPU so training does not keep two full
    SwinUNETR replicas on a 6GB GPU. For validation, use `eval_context()` to
    temporarily move EMA weights to the eval device.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: str | torch.device = "cpu",
    ) -> None:
        self.decay = float(decay)
        self.device = torch.device(device)
        # Deepcopy then move off GPU immediately to free VRAM.
        self.ema_model = deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        # Read live weights (may be on CUDA); write into CPU EMA buffers.
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            src = model_p.detach()
            if src.device != ema_p.device:
                src = src.to(ema_p.device, non_blocking=False)
            if torch.is_floating_point(ema_p):
                ema_p.mul_(d).add_(src, alpha=1.0 - d)
            else:
                ema_p.copy_(src)
        for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
            src = model_b.detach()
            if src.device != ema_b.device:
                src = src.to(ema_b.device, non_blocking=False)
            ema_b.copy_(src)

    def state_dict(self) -> dict[str, Any]:
        # Always CPU tensors for compact, portable checkpoints.
        cpu_sd = {k: v.detach().cpu() for k, v in self.ema_model.state_dict().items()}
        return {
            "decay": self.decay,
            "device": str(self.device),
            "ema_state_dict": cpu_sd,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "decay" in state:
            self.decay = float(state["decay"])
        if "ema_state_dict" in state:
            self.ema_model.load_state_dict(state["ema_state_dict"], strict=True)
            self.ema_model.to(self.device)

    def module_for_eval(self) -> nn.Module:
        """Return EMA module (may be on CPU). Prefer `eval_context` for GPU val."""
        return self.ema_model

    @contextmanager
    def eval_context(self, device: torch.device) -> Iterator[nn.Module]:
        """
        Temporarily move EMA weights to `device` for validation, then return to CPU.
        """
        target = torch.device(device)
        was = next(self.ema_model.parameters()).device
        try:
            if was != target:
                self.ema_model.to(target)
            self.ema_model.eval()
            yield self.ema_model
        finally:
            if self.device != target:
                self.ema_model.to(self.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

"""Centralized, environment-aware runtime configuration for the API layer.

Anything here that a client must not be able to override (which checkpoint
runs inference, which directories are reachable) lives as a server-side
constant. Only deployment-time knobs (CORS origins) read from the
environment, with safe local-dev defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The checkpoint used for all uploaded-case predictions. Clients never choose
# this — it is a deployment decision, not a request parameter.
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "exp_swinunetr_4class_et_fixed"
    / "checkpoints"
    / "best_mean_dice.pt"
)

# Roots that user-supplied paths are allowed to resolve within. Anything
# outside these directories is rejected before it ever reaches disk I/O or
# `torch.load`.
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
PREDICTIONS_ROOT = OUTPUTS_ROOT / "predictions"
UPLOADS_ROOT = PROJECT_ROOT / "uploads"
DATA_ROOTS = (PROJECT_ROOT / "data", UPLOADS_ROOT)

# Retention for generated artifacts. Both directories grow without bound
# otherwise — these are swept opportunistically (on each new session/job
# creation) rather than via a separate scheduler process.
PREDICTIONS_RETENTION_SECONDS = int(
    os.environ.get("PREDICTIONS_RETENTION_SECONDS", str(7 * 24 * 60 * 60))  # 7 days
)
# Upload sessions are normally deleted right after each job finishes; this is
# only a safety net for sessions orphaned by a crashed/killed job.
UPLOADS_RETENTION_SECONDS = int(
    os.environ.get("UPLOADS_RETENTION_SECONDS", str(60 * 60))  # 1 hour
)

_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]


def _load_cors_origins() -> list[str]:
    """Read allowed CORS origins from the environment, falling back to local dev ports.

    Set `CORS_ALLOWED_ORIGINS` as a comma-separated list in production,
    e.g. `CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com`.
    """
    raw = os.environ.get("CORS_ALLOWED_ORIGINS")
    if not raw:
        return list(_DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS_ALLOWED_ORIGINS = _load_cors_origins()

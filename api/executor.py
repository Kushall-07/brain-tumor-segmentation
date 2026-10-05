"""Bounded background execution for prediction jobs.

Each upload previously spawned its own raw `threading.Thread`, so N
concurrent uploads meant N inference threads fighting over the same GPU/VRAM
with no limit. This module provides a single shared thread pool with a
small, fixed size so prediction jobs queue instead of piling up unbounded.

A plain ThreadPoolExecutor is sufficient here — this is a single-process
deployment, not a multi-worker service, so a full task queue (Celery/RQ)
would be more infrastructure than the problem needs right now.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

# Inference runs on a single GPU, so concurrent model calls would thrash VRAM
# rather than run faster. Override with INFERENCE_MAX_WORKERS if deploying on
# a machine with more GPU headroom (or CPU-only, where some concurrency may
# still be safe).
INFERENCE_MAX_WORKERS = int(os.environ.get("INFERENCE_MAX_WORKERS", "1"))

_executor = ThreadPoolExecutor(
    max_workers=INFERENCE_MAX_WORKERS,
    thread_name_prefix="prediction-job",
)


def get_executor() -> ThreadPoolExecutor:
    """Return the shared bounded executor used for background prediction jobs."""
    return _executor

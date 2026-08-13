"""In-memory prediction job state management.

This module provides a simple thread-safe job store suitable for the current
single-server/local deployment of this academic project. For a production
multi-worker deployment, persistent or distributed job storage (e.g. Redis)
would be required so that job state survives process restarts and is visible
across workers.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from typing import Any

_lock = threading.Lock()

# job_id -> job state dict
_jobs: dict[str, dict[str, Any]] = {}


def create_job() -> str:
    """Create a new prediction job and return its unique ID."""
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "stage": "upload_received",
            "message": "MRI volumes received",
            "result": None,
            "error": None,
            "started_at": time.time(),
            "completed_at": None,
        }
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return a snapshot of job state, or None if the job does not exist."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return copy.deepcopy(job)


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    message: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> bool:
    """Update fields on an existing job. Returns False if job_id is unknown."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if status is not None:
            job["status"] = status
        if stage is not None:
            job["stage"] = stage
        if message is not None:
            job["message"] = message
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        return True


def complete_job(job_id: str, result: dict[str, Any]) -> None:
    """Mark a job as successfully completed with its prediction result."""
    final_result = dict(result)
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["completed_at"] = time.time()
            if job.get("started_at"):
                final_result["job_timing"] = {
                    "total_s": round(job["completed_at"] - job["started_at"], 3),
                }
    update_job(
        job_id,
        status="completed",
        stage="completed",
        message="Analysis completed successfully",
        result=final_result,
        error=None,
    )


def fail_job(job_id: str, error_message: str) -> None:
    """Mark a job as failed with a user-friendly error message."""
    update_job(
        job_id,
        status="failed",
        stage="failed",
        message="Analysis failed",
        result=None,
        error=error_message,
    )

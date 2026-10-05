"""Tests for the in-memory prediction job lifecycle (api/jobs.py), including
the opportunistic TTL-based pruning that stands in for a cleanup scheduler."""

import time

import api.jobs as jobs_module
from api.jobs import complete_job, create_job, fail_job, get_job, update_job


def setup_function(_):
    # Each test starts from an empty store so tests don't see jobs created
    # by other tests (the module holds a single process-wide dict).
    jobs_module._jobs.clear()


def test_create_job_starts_in_processing_state():
    job_id = create_job()
    job = get_job(job_id)
    assert job["status"] == "processing"
    assert job["stage"] == "upload_received"
    assert job["result"] is None
    assert job["error"] is None


def test_get_job_returns_none_for_unknown_id():
    assert get_job("does-not-exist") is None


def test_get_job_returns_a_copy_not_the_live_dict():
    job_id = create_job()
    snapshot = get_job(job_id)
    snapshot["status"] = "tampered"
    assert get_job(job_id)["status"] == "processing"


def test_update_job_changes_stage_and_message():
    job_id = create_job()
    assert update_job(job_id, stage="validation", message="Validating volumes") is True

    job = get_job(job_id)
    assert job["stage"] == "validation"
    assert job["message"] == "Validating volumes"
    assert job["status"] == "processing"  # untouched fields stay as-is


def test_update_job_returns_false_for_unknown_id():
    assert update_job("does-not-exist", stage="validation") is False


def test_complete_job_sets_result_status_and_timing():
    job_id = create_job()
    complete_job(job_id, {"mask_path": "outputs/predictions/x/mask.nii.gz"})

    job = get_job(job_id)
    assert job["status"] == "completed"
    assert job["stage"] == "completed"
    assert job["error"] is None
    assert job["result"]["mask_path"] == "outputs/predictions/x/mask.nii.gz"
    assert "job_timing" in job["result"]
    assert job["result"]["job_timing"]["total_s"] >= 0


def test_fail_job_sets_error_and_clears_result():
    job_id = create_job()
    fail_job(job_id, "Inference execution failed")

    job = get_job(job_id)
    assert job["status"] == "failed"
    assert job["stage"] == "failed"
    assert job["error"] == "Inference execution failed"
    assert job["result"] is None


def test_stale_finished_jobs_are_pruned_when_a_new_job_is_created():
    old_job_id = create_job()
    complete_job(old_job_id, {})
    # Simulate the job having finished well past the TTL.
    jobs_module._jobs[old_job_id]["completed_at"] = (
        time.time() - jobs_module.JOB_TTL_SECONDS - 1
    )

    create_job()  # create_job() opportunistically sweeps stale jobs

    assert get_job(old_job_id) is None


def test_recently_finished_jobs_are_not_pruned():
    recent_job_id = create_job()
    complete_job(recent_job_id, {})

    create_job()

    assert get_job(recent_job_id) is not None


def test_in_progress_jobs_are_never_pruned_regardless_of_age():
    in_progress_id = create_job()
    jobs_module._jobs[in_progress_id]["started_at"] = (
        time.time() - jobs_module.JOB_TTL_SECONDS * 10
    )

    create_job()

    assert get_job(in_progress_id) is not None

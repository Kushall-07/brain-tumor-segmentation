"""Tests for the HTTP-boundary path/checkpoint restrictions added to api/routes.py.

These exercise the actual FastAPI routes through TestClient rather than the
security helpers directly (test_security.py covers those in isolation), so
they catch regressions in how each endpoint wires resolve_within/
resolve_within_any into its request handling. Each test monkeypatches the
relevant *_ROOT constant on api.routes so it never touches the real
outputs/predictions directory.
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.routes as routes_module
from api.main import app

client = TestClient(app)


def test_download_serves_file_inside_predictions_root(tmp_path, monkeypatch):
    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    monkeypatch.setattr(routes_module, "PREDICTIONS_ROOT", predictions_root)

    case_dir = predictions_root / "prediction_x"
    case_dir.mkdir()
    (case_dir / "mask.nii.gz").write_bytes(b"fake-nifti-bytes")

    response = client.get("/download/prediction_x/mask.nii.gz")

    assert response.status_code == 200
    assert response.content == b"fake-nifti-bytes"


def test_download_rejects_absolute_path_outside_predictions_root(tmp_path, monkeypatch):
    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    monkeypatch.setattr(routes_module, "PREDICTIONS_ROOT", predictions_root)

    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret")

    response = client.get(f"/download/{outside_file}")

    assert response.status_code == 403


def test_download_rejects_relative_traversal_outside_predictions_root(tmp_path, monkeypatch):
    # httpx (used by TestClient) normalizes ".." out of the URL before the
    # request is even sent, so going through TestClient here would only test
    # httpx's normalization, not our own path-escape check. Call the route
    # handler directly instead, with the literal raw value a less well-behaved
    # HTTP client could still deliver.
    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    monkeypatch.setattr(routes_module, "PREDICTIONS_ROOT", predictions_root)

    # Sibling of predictions_root the request tries to reach via "..".
    (tmp_path / "secret.txt").write_text("secret")

    with pytest.raises(HTTPException) as exc_info:
        routes_module.download_prediction("sub/../../secret.txt")

    assert exc_info.value.status_code == 403


def test_download_returns_404_for_missing_file(tmp_path, monkeypatch):
    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    monkeypatch.setattr(routes_module, "PREDICTIONS_ROOT", predictions_root)

    response = client.get("/download/does-not-exist.nii.gz")

    assert response.status_code == 404


def test_class_analysis_rejects_mask_path_outside_predictions_root(tmp_path, monkeypatch):
    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    monkeypatch.setattr(routes_module, "PREDICTIONS_ROOT", predictions_root)

    outside_file = tmp_path / "secret.nii.gz"
    outside_file.write_bytes(b"not-really-nifti")

    response = client.post(
        "/predict/class-analysis",
        json={"mask_path": str(outside_file), "classes": [1, 2, 3]},
    )

    assert response.status_code == 403


def test_individual_class_analysis_rejects_mask_path_outside_predictions_root(
    tmp_path, monkeypatch
):
    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    monkeypatch.setattr(routes_module, "PREDICTIONS_ROOT", predictions_root)

    outside_file = tmp_path / "secret.nii.gz"
    outside_file.write_bytes(b"not-really-nifti")

    response = client.post(
        "/predict/individual-class-analysis",
        json={"mask_path": str(outside_file)},
    )

    assert response.status_code == 403


def test_research_model_info_rejects_checkpoint_outside_outputs_root(tmp_path, monkeypatch):
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir()
    monkeypatch.setattr(routes_module, "OUTPUTS_ROOT", outputs_root)

    outside_checkpoint = tmp_path / "evil.pt"
    outside_checkpoint.write_bytes(b"not a real checkpoint")

    response = client.get(
        "/research/model-info", params={"checkpoint_path": str(outside_checkpoint)}
    )

    assert response.status_code == 400


def test_research_model_info_accepts_missing_checkpoint_path():
    # No checkpoint_path at all is valid — falls back to static model metadata.
    response = client.get("/research/model-info")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_predict_rejects_checkpoint_path_outside_outputs_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir()
    monkeypatch.setattr(routes_module, "DATA_ROOTS", (data_root,))
    monkeypatch.setattr(routes_module, "OUTPUTS_ROOT", outputs_root)

    response = client.post(
        "/predict",
        json={
            "data_dir": str(data_root),
            "checkpoint_path": "/etc/passwd",
            "output_dir": str(outputs_root / "out"),
        },
    )

    assert response.status_code == 400
    assert "outside allowed directory" in response.json()["detail"]


def test_predict_rejects_data_dir_outside_allowed_roots(tmp_path, monkeypatch):
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir()
    monkeypatch.setattr(routes_module, "DATA_ROOTS", (tmp_path / "data",))
    monkeypatch.setattr(routes_module, "OUTPUTS_ROOT", outputs_root)

    response = client.post(
        "/predict",
        json={
            "data_dir": str(tmp_path / "somewhere-else"),
            "checkpoint_path": str(outputs_root / "model.pt"),
            "output_dir": str(outputs_root / "out"),
        },
    )

    assert response.status_code == 400
    assert "outside" in response.json()["detail"]


def test_predict_rejects_output_dir_outside_outputs_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir()
    monkeypatch.setattr(routes_module, "DATA_ROOTS", (data_root,))
    monkeypatch.setattr(routes_module, "OUTPUTS_ROOT", outputs_root)

    response = client.post(
        "/predict",
        json={
            "data_dir": str(data_root),
            "checkpoint_path": str(outputs_root / "model.pt"),
            "output_dir": str(tmp_path / "somewhere-else"),
        },
    )

    assert response.status_code == 400


def test_predict_upload_does_not_accept_checkpoint_path_query_param():
    # checkpoint_path must no longer be a client-controllable parameter at all.
    response = client.post(
        "/predict/upload",
        params={"checkpoint_path": "/etc/passwd", "save_probabilities": False},
    )
    # Missing required file parts -> 422, but critically this must not be a
    # route that silently accepts/forwards a client-supplied checkpoint path.
    assert response.status_code == 422
    body = response.json()
    assert "checkpoint_path" not in str(body)

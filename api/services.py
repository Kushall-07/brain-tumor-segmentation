from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

from api.exceptions import CheckpointError, InferenceError, ValidationError
from api.jobs import complete_job, create_job, fail_job, update_job
from api.schemas import MODALITY_FLAIR, MODALITY_T1, MODALITY_T1CE, MODALITY_T2, ModalityPaths
from api.utils import (
    cleanup_upload_session,
    create_prediction_dir,
    create_upload_session,
    save_modalities,
    save_ground_truth,
    calculate_tumor_volume,
    calculate_tumor_dimensions,
    calculate_tumor_volumes_4class,
    generate_class_specific_masks,
)
from api.validators import UploadValidator
from inference.predict import predict_case, predict_case_explicit
from utils.segmentation_evaluation import (
    evaluate_segmentation_masks,
    generate_comparison_mask,
)

logger = logging.getLogger(__name__)


def _is_uploaded_file(upload_file) -> bool:
    """Return True when an optional UploadFile part contains a real filename."""
    if upload_file is None:
        return False
    filename = getattr(upload_file, "filename", None)
    return bool(filename and str(filename).strip())


def predict_case_service(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    case_index: int = 0,
    save_probabilities: bool = False,
) -> dict[str, str | None]:
    """Service layer for brain tumor segmentation inference (CLI/training pipeline).

    Args:
        data_dir: Root directory containing case subfolders
        checkpoint_path: Path to trained checkpoint (.pt)
        output_dir: Output directory for results
        case_index: Index of case in data_dir to run inference on
        save_probabilities: If True, save class probabilities as NIfTI

    Returns:
        Dictionary with keys: case_id, mask_path, probability_path

    Raises:
        CheckpointError: If checkpoint_path does not exist
        InferenceError: If inference fails for any reason
    """
    # Validate input paths
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise ValidationError(
            f"Data directory does not exist: {data_dir_path}",
            client_message="Data directory not found",
        )

    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise CheckpointError(
            f"Checkpoint file does not exist: {checkpoint_path_obj}",
            client_message="Checkpoint file not found",
        )

    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Call the inference function
    try:
        result = predict_case(
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            out_dir=output_dir,
            case_index=case_index,
            save_probs=save_probabilities,
        )
        return result
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise InferenceError(
            f"Inference failed: {str(e)}",
            client_message="Inference execution failed",
        ) from e


def _execute_upload_prediction(
    *,
    flair,
    t1,
    t1ce,
    t2,
    checkpoint_path: str | Path,
    save_probabilities: bool = False,
    upload_session: Path | None = None,
    modality_paths: ModalityPaths | None = None,
    ground_truth_path: Path | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    request_id: str | None = None,
) -> dict[str, str | float | None]:
    """Shared upload prediction pipeline used by sync and async callers.

    When upload_session and modality_paths are provided, file saving is skipped
    (files were already persisted during the initial request).
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    session_created_here = upload_session is None

    try:
        logger.info(f"[{request_id}] START upload inference service")

        # Validate checkpoint exists
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise CheckpointError(
                f"Checkpoint file does not exist: {checkpoint_path_obj}",
                client_message="Checkpoint file not found",
            )

        if upload_session is None or modality_paths is None:
            # Validate uploaded files (extension and size only)
            UploadValidator.validate_upload(t1, MODALITY_T1)
            UploadValidator.validate_upload(t1ce, MODALITY_T1CE)
            UploadValidator.validate_upload(t2, MODALITY_T2)
            UploadValidator.validate_upload(flair, MODALITY_FLAIR)

            # Check for duplicate uploads
            UploadValidator.validate_no_duplicate_uploads({
                MODALITY_T1: t1,
                MODALITY_T1CE: t1ce,
                MODALITY_T2: t2,
                MODALITY_FLAIR: flair,
            })

            logger.info(f"[{request_id}] Basic upload validation completed")

            # Create upload session
            upload_session = create_upload_session()
            logger.info(f"[{request_id}] Created upload session: {upload_session}")

            # Create patient folder inside upload session
            patient_folder = upload_session / "BraTS-Patient"
            patient_folder.mkdir(parents=True, exist_ok=True)

            # Save uploaded files with standardized filenames
            modality_paths = save_modalities(t1, t1ce, t2, flair, patient_folder)
            logger.info(f"[{request_id}] Saved modalities to: {patient_folder}")

        if progress_callback:
            progress_callback("validation", "Validating MRI volumes")

        # Validate NIfTI format and shapes
        UploadValidator.validate_nifti_file(modality_paths.t1)
        UploadValidator.validate_nifti_file(modality_paths.t1ce)
        UploadValidator.validate_nifti_file(modality_paths.t2)
        UploadValidator.validate_nifti_file(modality_paths.flair)

        UploadValidator.validate_modality_shapes({
            MODALITY_T1: modality_paths.t1,
            MODALITY_T1CE: modality_paths.t1ce,
            MODALITY_T2: modality_paths.t2,
            MODALITY_FLAIR: modality_paths.flair,
        })
        logger.info(f"[{request_id}] NIfTI validation completed")

        # Permanent prediction directory (survives upload cleanup)
        prediction_dir = create_prediction_dir()
        expected_mask = prediction_dir / "BraTS-Patient_pred.nii.gz"
        logger.info(f"[{request_id}] Saving prediction to: {expected_mask.as_posix()}")

        # Call inference with explicit paths (no filename discovery)
        result = predict_case_explicit(
            t1_path=modality_paths.t1,
            t1ce_path=modality_paths.t1ce,
            t2_path=modality_paths.t2,
            flair_path=modality_paths.flair,
            checkpoint_path=checkpoint_path,
            out_dir=prediction_dir,
            save_probs=save_probabilities,
            request_id=request_id,
            progress_callback=progress_callback,
        )

        mask_path = Path(result["mask_path"])
        if not mask_path.exists():
            raise InferenceError(
                "Prediction file was not created",
                client_message="Inference completed but output file was not generated",
            )

        logger.info(f"[{request_id}] Prediction verified successfully.")

        if progress_callback:
            progress_callback("volume_analysis", "Calculating tumor volumes and dimensions")

        # Calculate tumor metrics from the segmentation mask
        volume_metrics: dict[str, float | int | list[float]] | None = None
        dimension_metrics: dict[str, float] | None = None
        class_volumes: dict[str, float] | None = None
        try:
            volume_metrics = calculate_tumor_volume(mask_path)
            logger.info(
                f"[{request_id}] Tumor volume calculated: "
                f"{volume_metrics['tumor_volume_cm3']} cm³"
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"[{request_id}] Failed to calculate tumor volume: {str(e)}")

        try:
            dimension_metrics = calculate_tumor_dimensions(mask_path)
            logger.info(
                f"[{request_id}] Tumor dimensions calculated: "
                f"H={dimension_metrics['height_mm']}mm, W={dimension_metrics['width_mm']}mm, L={dimension_metrics['length_mm']}mm"
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"[{request_id}] Failed to calculate tumor dimensions: {str(e)}")

        try:
            class_volumes = calculate_tumor_volumes_4class(mask_path)
            logger.info(
                f"[{request_id}] 4-class volumes calculated: "
                f"WT={class_volumes['whole_tumor_cm3']}cm³, TC={class_volumes['tumor_core_cm3']}cm³, ET={class_volumes['enhancing_tumor_cm3']}cm³"
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"[{request_id}] Failed to calculate 4-class volumes: {str(e)}")

        if progress_callback:
            progress_callback(
                "preparing_results",
                "Preparing visualization and analysis results",
            )

        # Copy FLAIR file to prediction directory for NiiVue viewer
        # This must happen BEFORE cleanup_upload_session deletes the upload session
        flair_copy_path = prediction_dir / "BraTS-Patient_flair.nii.gz"
        shutil.copy2(modality_paths.flair, flair_copy_path)
        logger.info(f"[{request_id}] Copied FLAIR to: {flair_copy_path.as_posix()}")

        # Normalize paths for stable API responses
        result["mask_path"] = mask_path.as_posix()
        result["mri_path"] = flair_copy_path.as_posix()
        if result.get("probability_path"):
            result["probability_path"] = Path(result["probability_path"]).as_posix()

        # Add tumor metrics to API response
        if volume_metrics is not None:
            result["tumor_volume_cm3"] = volume_metrics["tumor_volume_cm3"]
        else:
            result["tumor_volume_cm3"] = None

        if dimension_metrics is not None:
            result["dimensions_mm"] = dimension_metrics
        else:
            result["dimensions_mm"] = None

        if class_volumes is not None:
            result["volumes"] = class_volumes
        else:
            result["volumes"] = None

        # Generate class-specific visualization masks
        try:
            class_mask_paths = generate_class_specific_masks(mask_path, prediction_dir)
            result["class_masks"] = {
                "ncr_net": class_mask_paths[1].as_posix(),
                "edema": class_mask_paths[2].as_posix(),
                "et": class_mask_paths[3].as_posix(),
            }
            logger.info(f"[{request_id}] Generated class-specific visualization masks")
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to generate class-specific masks: {str(e)}")
            result["class_masks"] = None

        # Ground-truth evaluation (optional)
        result["ground_truth_available"] = False
        result["ground_truth_path"] = None
        result["validation_metrics"] = None
        result["validation_error"] = None
        result["validation_error_type"] = None
        result["comparison_mask_path"] = None
        result["comparison_error"] = None

        if ground_truth_path is not None and ground_truth_path.exists():
            gt_copy_path = prediction_dir / "BraTS-Patient_gt.nii.gz"
            shutil.copy2(ground_truth_path, gt_copy_path)
            result["ground_truth_path"] = gt_copy_path.as_posix()
            result["ground_truth_available"] = True

            logger.info("[%s] Ground-truth saved for evaluation: %s", request_id, gt_copy_path.as_posix())
            logger.info("[%s] Prediction mask for evaluation: %s", request_id, mask_path.as_posix())
            print(f"[VALIDATION] case ID: {result.get('case_id', request_id)}", flush=True)
            print(f"[VALIDATION] ground truth filename: {gt_copy_path.name}", flush=True)
            print(f"[VALIDATION] prediction filename: {mask_path.name}", flush=True)

            try:
                validation_result = evaluate_segmentation_masks(mask_path, gt_copy_path)
                if validation_result.get("available"):
                    result["validation_metrics"] = validation_result
                    logger.info("[%s] Ground-truth validation metrics computed", request_id)
                else:
                    result["validation_metrics"] = None
                    result["validation_error"] = validation_result.get("reason", "Validation failed")
                    result["validation_error_type"] = validation_result.get("error_type")
                    logger.warning(
                        "[%s] Ground-truth metric computation failed: %s (%s)",
                        request_id,
                        result["validation_error"],
                        result.get("validation_error_type"),
                    )
            except Exception as e:
                logger.warning("[%s] Ground-truth metric computation failed: %s", request_id, e, exc_info=True)
                result["validation_metrics"] = None
                result["validation_error"] = str(e)
                result["validation_error_type"] = type(e).__name__

            try:
                comparison_path = prediction_dir / "BraTS-Patient_comparison.nii.gz"
                generate_comparison_mask(
                    mask_path,
                    gt_copy_path,
                    comparison_path,
                    modality_paths.flair if modality_paths else flair_copy_path,
                )
                result["comparison_mask_path"] = comparison_path.as_posix()
                logger.info("[%s] Comparison mask generated: %s", request_id, comparison_path.as_posix())
            except Exception as e:
                logger.warning("[%s] Comparison mask generation failed: %s", request_id, e, exc_info=True)
                result["comparison_error"] = str(e)

        metadata: dict[str, str | float | int | list[float] | dict | None] = {
            "case_id": result.get("case_id"),
            "mask_path": result["mask_path"],
            "mri_path": result["mri_path"],
            "probability_path": result.get("probability_path"),
            "request_id": request_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tumor_volume_cm3": result["tumor_volume_cm3"],
        }
        if volume_metrics is not None:
            metadata["tumor_volume_mm3"] = volume_metrics["tumor_volume_mm3"]
            metadata["tumor_voxel_count"] = volume_metrics["tumor_voxel_count"]
            metadata["voxel_spacing_mm"] = volume_metrics["voxel_spacing_mm"]
            metadata["voxel_volume_mm3"] = volume_metrics["voxel_volume_mm3"]
        if dimension_metrics is not None:
            metadata["dimensions_mm"] = dimension_metrics
        if class_volumes is not None:
            metadata["volumes"] = class_volumes
        if result.get("timing"):
            metadata["timing"] = result["timing"]
        metadata["ground_truth_available"] = bool(result.get("ground_truth_available"))
        if result.get("ground_truth_path"):
            metadata["ground_truth_path"] = result["ground_truth_path"]
        if result.get("validation_metrics"):
            metadata["validation_metrics"] = result["validation_metrics"]
        if result.get("validation_error"):
            metadata["validation_error"] = result["validation_error"]
        if result.get("validation_error_type"):
            metadata["validation_error_type"] = result["validation_error_type"]
        if result.get("comparison_mask_path"):
            metadata["comparison_mask_path"] = result["comparison_mask_path"]
        if result.get("comparison_error"):
            metadata["comparison_error"] = result["comparison_error"]

        with (prediction_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[{request_id}] END upload inference service completed successfully")
        return result

    except (ValidationError, CheckpointError, InferenceError) as e:
        logger.error(f"[{request_id}] Request failed: {str(e)}", exc_info=True)
        raise  # Re-raise domain-specific exceptions

    except Exception as e:
        logger.error(f"[{request_id}] Upload inference failed: {str(e)}", exc_info=True)
        raise InferenceError(
            f"Inference failed: {str(e)}",
            client_message="Inference execution failed",
        ) from e

    finally:
        # Delete temporary uploads only — never delete outputs/predictions/
        if session_created_here and upload_session is not None:
            cleanup_upload_session(upload_session)
            logger.info(f"[{request_id}] Temporary upload session cleaned.")


def validate_case_metrics(
    prediction_mask_path: str | Path,
    ground_truth_mask_path: str | Path,
) -> dict[str, Any]:
    """Resolve mask paths and compute validation metrics for an existing case."""
    pred_resolved = _resolve_prediction_mask_path(prediction_mask_path)
    gt_resolved = _resolve_prediction_mask_path(ground_truth_mask_path)

    print("[VALIDATION] case validation request", flush=True)
    print(f"[VALIDATION] prediction filename: {pred_resolved.name}", flush=True)
    print(f"[VALIDATION] ground truth filename: {gt_resolved.name}", flush=True)

    return evaluate_segmentation_masks(pred_resolved, gt_resolved)


def _resolve_prediction_mask_path(mask_path: str | Path) -> Path:
    """Resolve a mask path to an absolute path within outputs/predictions when possible."""
    mask_path_obj = Path(mask_path)
    predictions_dir = Path("outputs/predictions").resolve()

    if mask_path_obj.is_absolute():
        requested_path = mask_path_obj.resolve()
    else:
        normalized = str(mask_path).replace("\\", "/")
        if "outputs/predictions/" in normalized:
            relative = normalized.split("outputs/predictions/", 1)[1]
            requested_path = (predictions_dir / relative).resolve()
        else:
            requested_path = (predictions_dir / mask_path_obj).resolve()

    try:
        requested_path.relative_to(predictions_dir)
    except ValueError as exc:
        raise ValueError(
            f"Mask path is outside predictions directory: {requested_path}"
        ) from exc

    if not requested_path.exists():
        raise FileNotFoundError(f"Mask file not found: {requested_path}")

    if not requested_path.is_file():
        raise ValueError(f"Path is not a file: {requested_path}")

    return requested_path


def predict_case_upload_service(
    flair,
    t1,
    t1ce,
    t2,
    checkpoint_path: str | Path,
    save_probabilities: bool = False,
) -> dict[str, str | None]:
    """Service layer for brain tumor segmentation inference with file uploads (API pipeline).

    Args:
        flair: Uploaded FLAIR MRI file
        t1: Uploaded T1 MRI file
        t1ce: Uploaded T1ce MRI file
        t2: Uploaded T2 MRI file
        checkpoint_path: Path to trained checkpoint (.pt)
        save_probabilities: If True, save class probabilities as NIfTI

    Returns:
        Dictionary with keys: case_id, mask_path, mri_path, probability_path

    Raises:
        ValidationError: If file validation fails
        CheckpointError: If checkpoint_path does not exist
        InferenceError: If inference fails for any reason
    """
    request_id = str(uuid.uuid4())[:8]
    return _execute_upload_prediction(
        flair=flair,
        t1=t1,
        t1ce=t1ce,
        t2=t2,
        checkpoint_path=checkpoint_path,
        save_probabilities=save_probabilities,
        request_id=request_id,
    )


def _run_prediction_job(
    job_id: str,
    upload_session: Path,
    modality_paths: ModalityPaths,
    checkpoint_path: str | Path,
    save_probabilities: bool,
    ground_truth_path: Path | None = None,
) -> None:
    """Background worker that runs the full prediction pipeline for a job."""

    def progress_callback(stage: str, message: str) -> None:
        update_job(job_id, stage=stage, message=message)

    request_id = job_id[:8]

    try:
        result = _execute_upload_prediction(
            flair=None,
            t1=None,
            t1ce=None,
            t2=None,
            checkpoint_path=checkpoint_path,
            save_probabilities=save_probabilities,
            upload_session=upload_session,
            modality_paths=modality_paths,
            ground_truth_path=ground_truth_path,
            progress_callback=progress_callback,
            request_id=request_id,
        )
        complete_job(job_id, result)
    except ValidationError as e:
        fail_job(job_id, e.client_message)
    except CheckpointError as e:
        fail_job(job_id, e.client_message)
    except InferenceError as e:
        fail_job(job_id, e.client_message)
    except Exception as e:
        logger.error(f"[{request_id}] Background job failed: {str(e)}", exc_info=True)
        fail_job(job_id, "Inference execution failed")
    finally:
        cleanup_upload_session(upload_session)
        logger.info(f"[{request_id}] Temporary upload session cleaned (background job).")


def start_prediction_job(
    flair,
    t1,
    t1ce,
    t2,
    checkpoint_path: str | Path,
    save_probabilities: bool = False,
    seg=None,
) -> dict[str, str]:
    """Accept uploads, persist files, create a job, and start background inference.

    Uploaded files are saved to disk before returning so they remain available
    after the HTTP request completes and UploadFile handles are closed.

    Returns:
        Dictionary with job_id and status for immediate client polling.
    """
    # Validate checkpoint exists before accepting the job
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise CheckpointError(
            f"Checkpoint file does not exist: {checkpoint_path_obj}",
            client_message="Checkpoint file not found",
        )

    # Basic upload validation (extension, size, duplicates)
    UploadValidator.validate_upload(t1, MODALITY_T1)
    UploadValidator.validate_upload(t1ce, MODALITY_T1CE)
    UploadValidator.validate_upload(t2, MODALITY_T2)
    UploadValidator.validate_upload(flair, MODALITY_FLAIR)
    UploadValidator.validate_no_duplicate_uploads({
        MODALITY_T1: t1,
        MODALITY_T1CE: t1ce,
        MODALITY_T2: t2,
        MODALITY_FLAIR: flair,
    })

    # Persist uploaded files before the request returns
    upload_session = create_upload_session()
    patient_folder = upload_session / "BraTS-Patient"
    patient_folder.mkdir(parents=True, exist_ok=True)
    modality_paths = save_modalities(t1, t1ce, t2, flair, patient_folder)

    ground_truth_path: Path | None = None
    if _is_uploaded_file(seg):
        from api.schemas import MODALITY_SEG

        logger.info("[GT] Saving ground-truth segmentation: %s", seg.filename)
        UploadValidator.validate_upload(seg, MODALITY_SEG)
        ground_truth_path = save_ground_truth(seg, patient_folder)
        UploadValidator.validate_nifti_file(ground_truth_path)
        logger.info("[GT] Ground truth saved to: %s", ground_truth_path.as_posix())
    else:
        logger.info("[GT] No ground-truth segmentation provided with this upload")

    job_id = create_job()

    thread = threading.Thread(
        target=_run_prediction_job,
        args=(
            job_id,
            upload_session,
            modality_paths,
            checkpoint_path,
            save_probabilities,
            ground_truth_path,
        ),
        daemon=True,
        name=f"prediction-job-{job_id[:8]}",
    )
    thread.start()

    logger.info(f"[{job_id[:8]}] Prediction job started: {job_id}")

    return {
        "job_id": job_id,
        "status": "processing",
    }

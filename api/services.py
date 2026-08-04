from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
import shutil

from api.exceptions import CheckpointError, InferenceError, ValidationError
from api.schemas import MODALITY_FLAIR, MODALITY_T1, MODALITY_T1CE, MODALITY_T2
from api.utils import (
    cleanup_upload_session,
    create_prediction_dir,
    create_upload_session,
    save_modalities,
    calculate_tumor_volume,
)
from api.validators import UploadValidator
from inference.predict import predict_case, predict_case_explicit

logger = logging.getLogger(__name__)


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
    upload_session = None
    
    try:
        logger.info(f"[{request_id}] START upload inference service")
        
        # Validate checkpoint exists
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise CheckpointError(
                f"Checkpoint file does not exist: {checkpoint_path_obj}",
                client_message="Checkpoint file not found",
            )

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
        
        logger.info(f"[{request_id}] VALIDATION completed")

        # Create upload session
        upload_session = create_upload_session()
        logger.info(f"[{request_id}] Created upload session: {upload_session}")

        # Create patient folder inside upload session
        patient_folder = upload_session / "BraTS-Patient"
        patient_folder.mkdir(parents=True, exist_ok=True)

        # Save uploaded files with standardized filenames
        modality_paths = save_modalities(t1, t1ce, t2, flair, patient_folder)
        logger.info(f"[{request_id}] Saved modalities to: {patient_folder}")

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
        )

        mask_path = Path(result["mask_path"])
        if not mask_path.exists():
            raise InferenceError(
                "Prediction file was not created",
                client_message="Inference completed but output file was not generated",
            )

        logger.info(f"[{request_id}] Prediction verified successfully.")

        # Calculate tumor volume from the segmentation mask
        try:
            volume_metrics = calculate_tumor_volume(mask_path)
            logger.info(f"[{request_id}] Tumor volume calculated: {volume_metrics['tumor_volume_cm3']} cm³")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"[{request_id}] Failed to calculate tumor volume: {str(e)}")
            # Set default values if calculation fails
            volume_metrics = {
                "tumor_voxel_count": 0,
                "voxel_spacing_mm": [0.0, 0.0, 0.0],
                "voxel_volume_mm3": 0.0,
                "tumor_volume_mm3": 0.0,
                "tumor_volume_cm3": 0.0,
            }

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
        
        # Add tumor volume to API response
        result["tumor_volume_cm3"] = volume_metrics["tumor_volume_cm3"]

        metadata = {
            "case_id": result.get("case_id"),
            "mask_path": result["mask_path"],
            "mri_path": result["mri_path"],
            "probability_path": result.get("probability_path"),
            "request_id": request_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tumor_volume_cm3": volume_metrics["tumor_volume_cm3"],
            "tumor_volume_mm3": volume_metrics["tumor_volume_mm3"],
            "tumor_voxel_count": volume_metrics["tumor_voxel_count"],
            "voxel_spacing_mm": volume_metrics["voxel_spacing_mm"],
            "voxel_volume_mm3": volume_metrics["voxel_volume_mm3"],
        }
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
        if upload_session is not None:
            cleanup_upload_session(upload_session)
            logger.info(f"[{request_id}] Temporary upload session cleaned.")

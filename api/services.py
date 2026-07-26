from __future__ import annotations

from pathlib import Path

from api.utils import cleanup_upload_session, create_upload_session, save_uploaded_file
from inference.predict import predict_case


def predict_case_service(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    case_index: int = 0,
    save_probabilities: bool = False,
) -> dict[str, str | None]:
    """Service layer for brain tumor segmentation inference.

    Args:
        data_dir: Root directory containing case subfolders
        checkpoint_path: Path to trained checkpoint (.pt)
        output_dir: Output directory for results
        case_index: Index of case in data_dir to run inference on
        save_probabilities: If True, save class probabilities as NIfTI

    Returns:
        Dictionary with keys: case_id, mask_path, probability_path

    Raises:
        FileNotFoundError: If data_dir or checkpoint_path does not exist
        RuntimeError: If inference fails for any reason
    """
    # Validate input paths
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir_path}")

    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path_obj}")

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
        raise RuntimeError(f"Inference failed: {str(e)}") from e


def predict_case_upload_service(
    flair,
    t1,
    t1ce,
    t2,
    checkpoint_path: str | Path,
    save_probabilities: bool = False,
) -> dict[str, str | None]:
    """Service layer for brain tumor segmentation inference with file uploads.

    Args:
        flair: Uploaded FLAIR MRI file
        t1: Uploaded T1 MRI file
        t1ce: Uploaded T1ce MRI file
        t2: Uploaded T2 MRI file
        checkpoint_path: Path to trained checkpoint (.pt)
        save_probabilities: If True, save class probabilities as NIfTI

    Returns:
        Dictionary with keys: case_id, mask_path, probability_path

    Raises:
        FileNotFoundError: If checkpoint_path does not exist
        RuntimeError: If inference fails for any reason
    """
    upload_session = None
    try:
        # Create upload session
        upload_session = create_upload_session()

        # Create patient folder inside upload session
        patient_folder = upload_session / "BraTS-Patient"
        patient_folder.mkdir(parents=True, exist_ok=True)

        # Save uploaded files preserving original filenames
        save_uploaded_file(flair, patient_folder)
        save_uploaded_file(t1, patient_folder)
        save_uploaded_file(t1ce, patient_folder)
        save_uploaded_file(t2, patient_folder)

        # Validate checkpoint exists
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path_obj}")

        # Create results directory
        results_dir = upload_session / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Call inference
        result = predict_case(
            data_dir=upload_session,
            checkpoint_path=checkpoint_path,
            out_dir=results_dir,
            case_index=0,
            save_probs=save_probabilities,
        )

        return result
    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}") from e
    finally:
        # Always cleanup upload session
        if upload_session is not None:
            cleanup_upload_session(upload_session)

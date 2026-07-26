from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Union

from fastapi import UploadFile

from api.schemas import ModalityPaths


def create_upload_session(base_dir: Union[str, Path] = "uploads") -> Path:
    """Create a unique upload session folder.

    Args:
        base_dir: Base directory for uploads (default: "uploads")

    Returns:
        Path to the newly created upload session folder
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    session_name = f"patient_{timestamp}_{unique_id}"

    session_path = base_path / session_name
    session_path.mkdir(parents=True, exist_ok=True)

    return session_path


def create_prediction_dir(base_dir: Union[str, Path] = "outputs/predictions") -> Path:
    """Create a unique permanent prediction output folder.

    Args:
        base_dir: Base directory for predictions (default: "outputs/predictions")

    Returns:
        Path to the newly created prediction folder
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    prediction_name = f"prediction_{timestamp}_{unique_id}"

    prediction_path = base_path / prediction_name
    prediction_path.mkdir(parents=True, exist_ok=True)

    return prediction_path


def save_uploaded_file(upload_file: UploadFile, destination: Union[str, Path]) -> Path:
    """Save an uploaded file to disk.

    Args:
        upload_file: FastAPI UploadFile object
        destination: Destination path (can be directory or full file path)

    Returns:
        Path to the saved file
    """
    dest_path = Path(destination)

    # If destination is a directory, use the original filename
    if dest_path.is_dir() or (not dest_path.suffix and not dest_path.exists()):
        dest_path = dest_path / upload_file.filename

    # Create parent directories if they don't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the file content
    with dest_path.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)

    # Reset file pointer for potential re-reading
    upload_file.file.seek(0)

    return dest_path


def cleanup_upload_session(folder: Union[str, Path]) -> None:
    """Delete an upload session folder recursively.

    Args:
        folder: Path to the upload session folder to delete
    """
    folder_path = Path(folder)
    if folder_path.exists():
        shutil.rmtree(folder_path, ignore_errors=True)


def save_modalities(
    t1: UploadFile,
    t1ce: UploadFile,
    t2: UploadFile,
    flair: UploadFile,
    patient_folder: Path,
) -> ModalityPaths:
    """Save all four MRI modalities preserving original file extensions.

    Args:
        t1: Uploaded T1 MRI file
        t1ce: Uploaded T1ce MRI file
        t2: Uploaded T2 MRI file
        flair: Uploaded FLAIR MRI file
        patient_folder: Directory to save the modalities

    Returns:
        ModalityPaths object with paths to saved files
    """
    def _get_extension(filename: str | None) -> str:
        if not filename:
            return ".nii.gz"
        fn_lower = filename.lower()
        if fn_lower.endswith(".nii.gz"):
            return ".nii.gz"
        if fn_lower.endswith(".nii"):
            return ".nii"
        return ".nii.gz"

    t1_ext = _get_extension(t1.filename)
    t1ce_ext = _get_extension(t1ce.filename)
    t2_ext = _get_extension(t2.filename)
    flair_ext = _get_extension(flair.filename)
    
    t1_path = save_uploaded_file(t1, patient_folder / f"t1{t1_ext}")
    t1ce_path = save_uploaded_file(t1ce, patient_folder / f"t1ce{t1ce_ext}")
    t2_path = save_uploaded_file(t2, patient_folder / f"t2{t2_ext}")
    flair_path = save_uploaded_file(flair, patient_folder / f"flair{flair_ext}")

    return ModalityPaths(
        t1=t1_path,
        t1ce=t1ce_path,
        t2=t2_path,
        flair=flair_path,
    )


